import streamlit as st
import pandas as pd
from pathlib import Path
from typing import Optional

from rag_audio_analysis.config import CYCLE_ANALYSIS_DIR
from rag_audio_analysis.source_bridge import (
    get_manual_units_for_session,
    get_rag_index_rows,
    build_doc_index_by_path,
    expand_transcript_context,
    build_manual_unit_index,
)
from rag_audio_analysis.settings import get_int
import json

st.set_page_config(page_title="Cycle Similarity Viewer", layout="wide")

st.title("Cycle similarity evidence viewer")
st.write("Browse `session_manual_similarity_evidence.csv` and `manual_unit_coverage_summary.csv` for a cycle.")


def list_cycles() -> list[str]:
    base = Path(CYCLE_ANALYSIS_DIR)
    if not base.exists():
        return []
    return sorted([p.name for p in base.iterdir() if p.is_dir()])


def load_csv_safe(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"Failed to read {path}: {e}")
        return None


cycles = list_cycles()
if not cycles:
    st.warning(f"No cycles found in {CYCLE_ANALYSIS_DIR}. Run the generator script first.")
    st.stop()

col1, col2 = st.columns([3, 1])
with col2:
    cycle = st.selectbox("Select cycle", options=cycles)
    refresh = st.button("Refresh cycles")
    if refresh:
        cycles = list_cycles()

cycle_dir = Path(CYCLE_ANALYSIS_DIR) / cycle
sim_path = cycle_dir / "session_manual_similarity_evidence.csv"
cov_path = cycle_dir / "manual_unit_coverage_summary.csv"

sim_df = load_csv_safe(sim_path)
cov_df = load_csv_safe(cov_path)

if sim_df is None and cov_df is None:
    st.error("No similarity evidence or coverage CSVs found for this cycle.")
    st.stop()

st.sidebar.header("Filters")
selected_session = st.sidebar.selectbox(
    "Session (manual_session_num)",
    options=["ALL"] + (sorted(sim_df["manual_session_num"].astype(str).unique().tolist()) if sim_df is not None else []),
)
manual_unit_input = st.sidebar.text_input("Manual unit id (exact match)")
min_similarity = st.sidebar.slider("Min similarity", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
max_rows = st.sidebar.number_input("Max rows to show", min_value=10, max_value=10000, value=500)

show_coverage = st.sidebar.checkbox("Show coverage summary", value=True)
show_session_scores = st.sidebar.checkbox("Compute session coverage scores", value=True)
recompute_counts = st.sidebar.button("Recompute manual unit counts")

# Filter evidence table
if sim_df is not None:
    df = sim_df.copy()
    # ensure numeric score field exists
    score_col = "mapped_manual_unit_match_score"
    if score_col in df.columns:
        try:
            df[score_col] = pd.to_numeric(df[score_col], errors="coerce").fillna(0.0)
        except Exception:
            df[score_col] = df[score_col]

    if selected_session != "ALL":
        df = df[df["manual_session_num"].astype(str) == str(selected_session)]
    if manual_unit_input:
        df = df[df["query_manual_unit_id"].astype(str) == manual_unit_input]
    if score_col in df.columns:
        df = df[df[score_col] >= float(min_similarity)]

    total_rows = len(df)
    st.subheader(f"Similarity evidence — {total_rows} rows")
    st.write(f"Source: {sim_path}")

    if total_rows == 0:
        st.info("No matching rows for the current filters.")
    else:
        st.dataframe(df.head(int(max_rows)))

        # download filtered CSV
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download filtered evidence CSV", csv_bytes, file_name=f"{cycle}_filtered_similarity.csv")

        # row selector + detail panel
        display_df = df.head(int(max_rows)).reset_index(drop=False)
        options: list[str] = []
        label_map: dict[str, int] = {}
        for pos, r in display_df.reset_index(drop=True).iterrows():
            # use positional index (pos) for stable selection and lookup
            orig_idx = int(r.get('index')) if 'index' in r else pos
            manual_session = r.get('manual_session_num', '')
            unit = r.get('query_manual_unit_id', '')
            subsection = r.get('query_manual_unit_subsection', '') or r.get('manual_subsection', '')
            score = r.get('mapped_manual_unit_match_score', r.get('manual_unit_match_score', ''))
            session_id = r.get('session_id', '')
            # keep a short excerpt for eyeballing
            excerpt = str(r.get('text_excerpt', '') or '').replace('\n', ' ').strip()[:60]
            label = f"{cycle}|s:{manual_session}|u:{unit}|sub:{subsection}|score:{score}|sid:{session_id}|{excerpt}"
            options.append(label)
            label_map[label] = int(pos)

        if options:
            choice = st.selectbox("Select evidence row to inspect", options=options)
            sel_pos = label_map.get(choice)
            if sel_pos is not None:
                selected = display_df.reset_index(drop=True).iloc[int(sel_pos)].to_dict()

                st.markdown("### Selected evidence details")
                # toggle for full text display (use unique key to avoid collisions)
                # default to True so full texts are visible immediately on selection
                show_full = st.checkbox("Show full query & full window text", value=True, key=f"show_full_{cycle}_{sel_pos}")

                # display metadata fields
                meta_cols = [c for c in selected.keys() if c not in ('text_excerpt','query_text_excerpt')]
                for k in meta_cols:
                    st.markdown(f"**{k}**: {selected.get(k, '')}")

                # show query (try to lookup full manual unit text)
                qid = selected.get('query_manual_unit_id', '')
                session_num = str(selected.get('manual_session_num', ''))
                full_query = ''
                try:
                    units = get_manual_units_for_session(session_num)
                    for u in units:
                        if str(u.get('manual_unit_id', '') or u.get('id', '')) == str(qid):
                            full_query = u.get('matching_text') or u.get('text') or u.get('manual_text') or ''
                            break
                except Exception:
                    full_query = ''

                if show_full:
                    st.markdown("**Full query text:**")
                    st.text_area("", value=str(full_query or selected.get('query_text_excerpt','')), height=240)
                else:
                    st.markdown("**Query excerpt:**")
                    st.write(selected.get('query_text_excerpt', ''))

                # show matched window full context (expand from meta rows if possible)
                doc_idx = selected.get('doc_index')
                window_full = ''
                try:
                    if doc_idx is not None and str(doc_idx).strip() != '':
                        meta_rows = get_rag_index_rows()
                        path_lookup = build_doc_index_by_path(meta_rows)
                        ctx = expand_transcript_context(int(doc_idx), meta_rows=meta_rows, path_lookup=path_lookup, window=int(get_int('transcript_export','context_window',2)))
                        window_full = ctx.get('text', '')
                except Exception:
                    window_full = ''

                if show_full:
                    st.markdown("**Full matched window text:**")
                    st.text_area("", value=str(window_full or selected.get('text_excerpt','')), height=360)
                else:
                    st.markdown("**Matched window excerpt:**")
                    st.write(selected.get('text_excerpt', ''))

# Coverage summary
if show_coverage and cov_df is not None:
    cov = cov_df.copy()
    if selected_session != "ALL":
        cov = cov[cov["manual_session_num"].astype(str) == str(selected_session)]
    if manual_unit_input:
        cov = cov[cov["query_manual_unit_id"].astype(str) == manual_unit_input]

    st.subheader(f"Manual unit coverage — {len(cov)} rows")
    st.write(f"Source: {cov_path}")
    if cov.empty:
        st.info("No coverage rows for current filters.")
    else:
        st.dataframe(cov)
        csv_cov = cov.to_csv(index=False).encode("utf-8")
        st.download_button("Download coverage CSV", csv_cov, file_name=f"{cycle}_coverage_filtered.csv")

# Session coverage scores (simple coverage = % of manual units with matched==True)
if show_session_scores and cov_df is not None:
    st.subheader("Session coverage scores (simple coverage)")
    # normalize matched column to boolean
    cov_work = cov_df.copy()
    if "matched" in cov_work.columns:
        cov_work["_matched_bool"] = cov_work["matched"].astype(str).str.lower().map(lambda v: True if v in ("1", "true", "yes") else False)
    else:
        cov_work["_matched_bool"] = False

    # group matched counts by manual_session_num (from coverage CSV)
    matched_df = (
        cov_work.groupby("manual_session_num").agg(matched_units=('_matched_bool', 'sum')).reset_index()
    )

    # load or compute canonical total unit counts per session and cache them under the cycle directory
    counts_path = cycle_dir / "manual_unit_counts.json"
    counts: dict = {}
    if counts_path.exists() and not recompute_counts:
        try:
            counts = json.loads(counts_path.read_text(encoding='utf-8'))
        except Exception:
            counts = {}

    # determine sessions we need counts for
    sessions_needed = [str(x) for x in matched_df['manual_session_num'].astype(str).unique().tolist()]

    # If recompute requested or counts missing, compute counts in bulk by building the manual unit index once.
    if recompute_counts or not counts:
        try:
            all_units = build_manual_unit_index()
            bulk_counts: dict = {}
            for u in all_units:
                sess = str(u.get('manual_week', '') or '').strip()
                if not sess:
                    continue
                bulk_counts[sess] = bulk_counts.get(sess, 0) + 1
            # ensure we have entries for sessions_needed even if zero
            for s in sessions_needed:
                counts[s] = int(bulk_counts.get(s, 0))
        except Exception:
            # fallback to per-session calls if bulk index isn't available
            for s in sessions_needed:
                if s in counts and isinstance(counts[s], int):
                    continue
                try:
                    units = get_manual_units_for_session(s)
                    counts[s] = int(len(units))
                except Exception:
                    counts[s] = 0
    else:
        # fill any missing session counts from per-session lookup (rare)
        for s in sessions_needed:
            if s in counts and isinstance(counts[s], int):
                continue
            try:
                units = get_manual_units_for_session(s)
                counts[s] = int(len(units))
            except Exception:
                counts[s] = 0

    # persist counts for future runs
    try:
        counts_path.write_text(json.dumps(counts, indent=2), encoding='utf-8')
    except Exception:
        pass

    # combine matched counts with canonical totals
    score_rows = []
    for _, row in matched_df.iterrows():
        s = str(row['manual_session_num'])
        matched_units = int(row['matched_units'])
        total_units = int(counts.get(s, 0))
        coverage_rate = float(matched_units) / total_units if total_units > 0 else 0.0
        score_rows.append({'manual_session_num': s, 'total_units': total_units, 'matched_units': matched_units, 'coverage_rate': coverage_rate})

    score_df = pd.DataFrame(score_rows)

    # filter if sidebar selection applied
    if selected_session != "ALL":
        score_df = score_df[score_df["manual_session_num"].astype(str) == str(selected_session)]
    if manual_unit_input:
        # if user filtered a specific unit, recompute stats for sessions that include that unit
        filtered = cov_work[cov_work["query_manual_unit_id"].astype(str) == manual_unit_input]
        score_df = (
            filtered.groupby("manual_session_num").agg(total_units=("query_manual_unit_id", "nunique"), matched_units=("_matched_bool", "sum")).reset_index()
        )
        score_df["coverage_rate"] = score_df.apply(lambda r: float(r["matched_units"]) / int(r["total_units"]) if int(r["total_units"]) > 0 else 0.0, axis=1)

    score_df = score_df.sort_values("coverage_rate", ascending=False)

    st.write("Coverage rate = matched_units / total_manual_units (per session)")
    st.dataframe(score_df)

    # bar chart
    try:
        st.bar_chart(score_df.set_index("manual_session_num")["coverage_rate"])
    except Exception:
        pass

    csv_scores = score_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download session coverage scores", csv_scores, file_name=f"{cycle}_session_coverage_scores.csv")

st.write("\n---\n")
st.caption("Built-in viewer for `session_manual_similarity_evidence.csv` and `manual_unit_coverage_summary.csv` — lightweight Streamlit UI.")
