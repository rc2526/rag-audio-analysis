"""Streamlit demo: RAG vs non-RAG chat UI (minimal skeleton).

Run with: streamlit run app/rag_demo.py
"""
from __future__ import annotations

import streamlit as st
from rag_audio_analysis.rag_service import retrieve_for_question, answer_rag, answer_non_rag
import json
from rag_audio_analysis.settings import get_int, get_float, get_list
from pathlib import Path

st.set_page_config(page_title="RAG Demo", layout="wide")

st.title("PMH Intervention Analysis - RAG Demo")

with st.sidebar:
    # Prefer cycles from settings.ini (section: cycle_analysis -> cycles). Falls back to a small
    # discovery of directories under data/derived/cycle_analysis if the settings value is empty.
    cfg_cycles = get_list("cycle_analysis", "cycles", ["1", "2", "3"])
    cycle_options = [f"PMHCycle{c}" if not str(c).startswith("PMHCycle") else str(c) for c in cfg_cycles]

    # discovery fallback helper
    def discover_cycles_from_disk() -> list[str]:
        base = Path("data/derived/cycle_analysis")
        if not base.exists():
            return []
        return sorted([p.name for p in base.iterdir() if p.is_dir()])

    if not cycle_options:
        cycle_options = discover_cycles_from_disk()

    # allow manual refresh if the config is stale
    if st.button("Refresh cycles"):
        discovered = discover_cycles_from_disk()
        if discovered:
            cycle_options = discovered

    if not cycle_options:
        cycle_options = ["PMHCycle1", "PMHCycle2", "PMHCycle3"]

    cycle = st.selectbox("Cycle", options=["All cycles"] + cycle_options, index=0)
    # map UI selection to cycle param used by retrieval; empty string means no cycle restriction
    cycle_param = "" if cycle == "All cycles" else cycle
    default_topk = get_int("chat", "topk", 8)
    # Prefer manual-unit similarity threshold by default so the demo returns manual evidence
    # unless the user explicitly lowers/raises the slider.
    default_min_sim = get_float("topic_matching", "manual_unit_min_similarity", 0.28)
    top_k = st.slider("top_k", min_value=1, max_value=20, value=default_topk)
    min_sim = st.slider("min_similarity", min_value=0.0, max_value=1.0, value=float(default_min_sim), step=0.01)
    prompt_variant = st.selectbox("prompt_variant", options=["default", "pi_question", "fidelity"], index=0)
    # Messaging UI toggle: when enabled, the app behaves like a messaging client that
    # preserves history and sends recent turns as context with each retrieval+answer call.
    messaging_ui = st.checkbox("Messaging-style UI (preserve history)", value=False)

if "history" not in st.session_state:
    st.session_state.history = []

def send_query(q: str):
    # use a small role/text contract so backends expecting 'role'/'text' can consume history
    st.session_state.history.append({"role": "user", "text": q})

if not messaging_ui:
    # legacy form-based interaction
    with st.form("query_form", clear_on_submit=True):
        q = st.text_input("Ask a question about the selected cycle transcripts:")
        submitted = st.form_submit_button("Send")
        if submitted and q:
            send_query(q)
            with st.spinner("Retrieving and answering..."):
                rag_out = answer_rag(q, cycle_param, top_k=top_k, min_similarity=min_sim, prompt_variant=prompt_variant)
                non_out = answer_non_rag(q)
            st.session_state.history[-1]["rag"] = rag_out
            st.session_state.history[-1]["nonrag"] = non_out
else:
    # Messaging-style composer: single-line input + send button; include history in the call
    col_msg, col_send = st.columns([8, 1])
    with col_msg:
        # initialize the key if missing
        if "messaging_input" not in st.session_state:
            st.session_state["messaging_input"] = ""
        st.text_input("Message:", key="messaging_input")

    def _messaging_send() -> None:
        msg_text = str(st.session_state.get("messaging_input", "") or "").strip()
        if not msg_text:
            return
        # append the user turn and call retrieval+answer; we perform these state updates inside the
        # callback so clearing the widget value is allowed.
        send_query(msg_text)
        with st.spinner("Retrieving and answering..."):
            rag_out = answer_rag(
                msg_text,
                cycle_param,
                top_k=top_k,
                min_similarity=min_sim,
                prompt_variant=prompt_variant,
                history=st.session_state.history,
                history_turns=2,
            )
        # append assistant turn so history contains both roles; include full rag payload for UI
        st.session_state.history.append({"role": "assistant", "text": str(rag_out.get("answer_raw", "")), "rag": rag_out})
        # clear composer
        st.session_state["messaging_input"] = ""

    with col_send:
        st.button("Send", key="messaging_send", on_click=_messaging_send)

if messaging_ui:
    st.header("Chat")
    # Render history in chronological order
    def _try_parse_json_prefix(s: str) -> dict | None:
        """Try to parse a leading JSON object from s. Returns dict or None."""
        if not isinstance(s, str):
            return None
        s = s.strip()
        try:
            return json.loads(s)
        except Exception:
            # attempt to extract first {...} block
            first = s.find("{")
            last = s.rfind("}")
            if first != -1 and last != -1 and last > first:
                sub = s[first : last + 1]
                try:
                    return json.loads(sub)
                except Exception:
                    return None
            return None
    for turn in st.session_state.history:
        # Support two history shapes: legacy {'user':...} or new {'role':'user'|'assistant','text':...}
        role = None
        text = ""
        if turn.get("role"):
            role = turn.get("role")
            text = str(turn.get("text", ""))
        elif turn.get("user"):
            role = "user"
            text = str(turn.get("user", ""))
        else:
            # unknown shape: render raw
            role = "user"
            text = str(turn)

        # User bubble aligned to the right
        u_left, u_right = st.columns([3, 7])
        with u_left:
            st.write("")
        with u_right:
            if role == "user":
                st.markdown(f"<div style='background:#e6f2ff;border-radius:8px;padding:8px;text-align:right'>**You:** {text}</div>", unsafe_allow_html=True)
            else:
                # if this is an assistant entry that was appended as a separate history item, we'll render below
                st.write("")

        # Assistant / RAG response (if present)
        # If this history item contains a rag payload, render assistant bubble and evidence now.
        if turn.get("rag"):
            rag = turn.get("rag", {})
            ans = rag.get("answer_raw", "")
            a_left, a_right = st.columns([7, 3])
            with a_left:
                # show assistant bubble
                if isinstance(ans, dict):
                    # pretty-print JSON answer
                    st.markdown("<div style='background:#f1f3f4;border-radius:8px;padding:8px'>**Assistant:**</div>", unsafe_allow_html=True)
                    st.json(ans)
                else:
                    st.markdown(f"<div style='background:#f1f3f4;border-radius:8px;padding:8px'>**Assistant:** {str(ans)}</div>", unsafe_allow_html=True)

                # Evidence expander
                windows = rag.get("windows", []) or []
                manuals = rag.get("manuals", []) or []
                n_evidence = len(windows) + len(manuals)
                with st.expander(f"Evidence ({n_evidence})", expanded=False):
                    if windows:
                        st.markdown("**Transcript windows**")
                        for w in windows:
                            st.markdown(f"- **[WIN]** score={w.get('score',0):.3f} — speaker={w.get('speaker','')} — path={w.get('path','')}")
                            st.code(w.get('text','(no text)'))
                    if manuals:
                        st.markdown("**Manual units**")
                        for m in manuals:
                            # prefer an explicit session number, fall back to transcript_id or manual_session if present
                            sess = m.get('session_num') or m.get('transcript_id') or m.get('manual_session') or ""
                            sess_label = f"(session={sess}) " if sess else ""
                            st.markdown(f"- **[MAN] {m.get('manual_unit_id','')}** {sess_label}(week={m.get('manual_week','')}) — score={m.get('score',0):.3f}")
                            # show the short chunk text and the richer matching_text (heading + chunk) for context
                            st.write(m.get('text',''))
                            matching = m.get('matching_text') or ""
                            if matching and matching.strip() != str(m.get('text','')).strip():
                                with st.expander("Matching text (heading + chunk)", expanded=False):
                                    st.text(matching)

                # Prompt inspector
                prompt_text = rag.get('prompt') or rag.get('prompt_text')
                if prompt_text:
                    with st.expander("Show prompt (for audit)"):
                        st.code(prompt_text)

            with a_right:
                st.write("")
    st.markdown("---")
else:
    # legacy two-column view
    col1, col2 = st.columns(2)

    with col1:
        st.header("RAG answer")
        for turn in st.session_state.history[::-1]:
            if "rag" not in turn:
                continue
            st.markdown(f"**User:** {turn['user']}")
            # show model answer
            st.json(turn["rag"]["answer_raw"]) if isinstance(turn["rag"]["answer_raw"], dict) else st.text(turn["rag"]["answer_raw"])

            # show human-readable retrieval payload (windows + manuals)
            out = turn["rag"]
            with st.expander("Retrieved evidence (windows + manuals)"):
                st.subheader("Transcript windows")
                for w in out.get("windows", []):
                    st.markdown(f"- **Score:** {w.get('score', 0):.3f} — **Speaker:** {w.get('speaker','')} — **Path:** {w.get('path','')}")
                    st.code(w.get("text", "(no text)"))
                st.subheader("Manual units")
                for m in out.get("manuals", []):
                    sess = m.get('session_num') or m.get('transcript_id') or m.get('manual_session') or ""
                    sess_label = f"(session={sess}) " if sess else ""
                    st.markdown(f"- **{m.get('manual_unit_id','')}** {sess_label}(week {m.get('manual_week','')}) — score {m.get('score',0):.3f}")
                    st.write(m.get("text", ""))
            st.markdown("---")

    with col2:
        st.header("Non-RAG answer")
        for turn in st.session_state.history[::-1]:
            if "nonrag" not in turn:
                continue
            st.markdown(f"**User:** {turn['user']}")
            st.json(turn["nonrag"]["answer_raw"]) if isinstance(turn["nonrag"]["answer_raw"], dict) else st.text(turn["nonrag"]["answer_raw"])
            st.markdown("---")

st.sidebar.markdown("---")
if st.sidebar.button("Clear history"):
    st.session_state.history = []
