## Results Template

### Automated Manual Alignment Across Cycles

Across cycles 1-5, the automated session-topic pipeline generated fidelity summaries for each available session-topic pair. Aggregate fidelity statistics were derived from `summary_fidelity_by_cycle.csv` and `summary_fidelity_by_topic.csv`. Mean adherence scores ranged from `[INSERT RANGE]`, with `[INSERT CYCLE]` showing the highest average adherence and `[INSERT CYCLE]` the lowest. The proportion of session-topic pairs classified as high adherence ranged from `[INSERT RANGE]%`, whereas low-adherence pairs accounted for `[INSERT RANGE]%` of outputs across cycles.

Manual-unit coverage and subsection coverage followed similar patterns. Across all cycles, the mean manual-unit coverage was `[INSERT VALUE]`, and the mean subsection coverage was `[INSERT VALUE]`. Topics with the highest average adherence included `[INSERT TOPICS]`, whereas topics with lower average adherence included `[INSERT TOPICS]`. These findings suggest that some intervention topics were more consistently reflected in transcript evidence than others.

### Automated Answers to PI Questions

The pipeline also generated question-specific summaries for four prespecified analytic questions: facilitator reference to the topic, facilitator demonstration of the skill, participant practice in session or at home, and participant use of the skill with their child at home. Aggregate question-level statistics were derived from `summary_pi_questions_by_cycle.csv`, `summary_pi_questions_by_type.csv`, and `summary_pi_questions_by_cycle_and_type.csv`.

Across all cycles, the mean number of retrieved evidence windows per question was `[INSERT VALUE]`. The percentage of question rows with non-empty model-generated answers was `[INSERT VALUE]%`, and the percentage of rows with explicit evidence references was `[INSERT VALUE]%`. Model-reported confidence was most often `[INSERT LOW/MEDIUM/HIGH]`, with `[INSERT QUESTION TYPE]` showing the highest proportion of high-confidence outputs and `[INSERT QUESTION TYPE]` showing the highest proportion of low-confidence outputs.

For facilitator reference and facilitator demonstration questions, the retrieved evidence commonly reflected structured teaching, review of session content, guided discussion, or skill modeling. For participant practice questions, outputs frequently described in-session reflection, self-monitoring, and assigned home practice. For participant-child-home questions, outputs more often indicated sparse or absent evidence, suggesting that child-directed home application may have been less explicitly discussed or less consistently captured in the available transcript windows.

### Representative Evidence

Representative evidence excerpts and question-linked transcript windows can be reviewed in `table_topic_evidence.csv` and `table_pi_question_answers.csv`. For example, `[INSERT EXAMPLE]` illustrated facilitator-led explanation of stress and eating, whereas `[INSERT EXAMPLE]` illustrated participant discussion of stress-related eating behavior. In cases where the child-home question returned no evidence, the associated retrieved windows did not contain explicit references to practicing the skill with a child at home.

### Suggested Tables and Figures

Use these outputs for manuscript tables:

- `table_session_topic_fidelity.csv`
  - detailed session-topic fidelity table
- `table_pi_question_answers.csv`
  - detailed question-level answer table
- `summary_fidelity_by_cycle.csv`
  - cycle-level fidelity statistics
- `summary_fidelity_by_topic.csv`
  - topic-level fidelity statistics
- `summary_pi_questions_by_cycle.csv`
  - cycle-level PI-question statistics
- `summary_pi_questions_by_type.csv`
  - question-type statistics
- `summary_pi_questions_by_cycle_and_type.csv`
  - question-type by cycle statistics
- `summary_evidence_by_cycle.csv`
  - evidence-score summary table

Suggested figures:

1. Boxplot of adherence score by cycle
2. Stacked bar chart of adherence labels by cycle
3. Heatmap of topic by cycle mean adherence
4. Bar chart of mean retrieved evidence count by PI question type
