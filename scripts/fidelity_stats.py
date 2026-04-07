import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path('data/derived/cycle_analysis/summary')
OUT = BASE / 'fidelity_analysis.txt'

# Files
session_fidelity_path = BASE / 'table_cycle_manual_session_fidelity.csv'
cycle_summary_path = BASE / 'summary_fidelity_by_cycle.csv'

# Read
sf = pd.read_csv(session_fidelity_path, dtype=str)
cs = pd.read_csv(cycle_summary_path, dtype=str)

# Helper to coerce numeric
for col in ['retrieved_evidence_count','expected_manual_unit_count','matched_manual_unit_count','manual_unit_coverage','subsection_coverage','adherence_score','evidence_density']:
    if col in sf.columns:
        sf[col] = pd.to_numeric(sf[col], errors='coerce')

# Global metrics
report_lines = []
report_lines.append('Fidelity analysis report')
report_lines.append('========================')
report_lines.append(f'Total session-topic rows: {len(sf):,}')

# Adherence score summary
if 'adherence_score' in sf.columns:
    report_lines.append('\nOverall adherence_score')
    report_lines.append(sf['adherence_score'].describe().to_string())

# Coverage summaries
if 'manual_unit_coverage' in sf.columns:
    report_lines.append('\nManual unit coverage (mean, median, std)')
    report_lines.append(f"mean={sf['manual_unit_coverage'].mean():.3f}, median={sf['manual_unit_coverage'].median():.3f}, std={sf['manual_unit_coverage'].std():.3f}")
if 'subsection_coverage' in sf.columns:
    report_lines.append('\nSubsection coverage (mean, median, std)')
    report_lines.append(f"mean={sf['subsection_coverage'].mean():.3f}, median={sf['subsection_coverage'].median():.3f}, std={sf['subsection_coverage'].std():.3f}")

# Adherence label distribution
if 'adherence_label' in sf.columns:
    dist = sf['adherence_label'].fillna('MISSING').value_counts(dropna=False)
    report_lines.append('\nAdherence label distribution:')
    report_lines.extend([f'  {idx}: {cnt}' for idx, cnt in dist.items()])

# Per-cycle stats
if 'cycle_id' in sf.columns:
    report_lines.append('\nPer-cycle adherence scores (mean, median, n)')
    grp = sf.groupby('cycle_id')
    for name, g in grp:
        if 'adherence_score' in g.columns:
            report_lines.append(f'  {name}: mean={g["adherence_score"].mean():.3f}, median={g["adherence_score"].median():.3f}, n={len(g)}')

# Top / bottom sessions by adherence_score
if 'adherence_score' in sf.columns:
    # handle optional session_label column
    label_col = 'session_label' if 'session_label' in sf.columns else None
    top = sf.nlargest(10, 'adherence_score')
    bot = sf.nsmallest(10, 'adherence_score')
    report_lines.append('\nTop 10 sessions by adherence_score:')
    for _, r in top.iterrows():
        label = r[label_col] if label_col else ''
        report_lines.append(f"  {r['cycle_id']} session {r['manual_session_num']} ({label}): {r['adherence_score']:.3f}")
    report_lines.append('\nBottom 10 sessions by adherence_score:')
    for _, r in bot.iterrows():
        label = r[label_col] if label_col else ''
        report_lines.append(f"  {r['cycle_id']} session {r['manual_session_num']} ({label}): {r['adherence_score']:.3f}")

# Correlations: retrieved evidence count vs adherence
if 'retrieved_evidence_count' in sf.columns and 'adherence_score' in sf.columns:
    df_corr = sf[['retrieved_evidence_count','adherence_score']].dropna()
    if len(df_corr) >= 3:
        corr = df_corr['retrieved_evidence_count'].corr(df_corr['adherence_score'])
        report_lines.append(f'\nCorrelation between retrieved_evidence_count and adherence_score: {corr:.3f}')

# Save and print
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text('\n'.join(report_lines), encoding='utf-8')
print('\n'.join(report_lines))
print(f"\nWrote report to: {OUT}")
