# Evaluation Sets

`sample_eval.jsonl` contains the Stage 6 starter evaluation set.

Each line is a JSON object with:

- `id`
- `question`
- `ground_truth`
- `expected_doc`
- `expected_keywords`
- `question_type`

The current version contains 25 samples covering sample Markdown, txt, sample PDF, the real CMD3 PDF, summary/method/compare/fact questions, confusing document-selection questions, and 3 unanswerable questions.

For unanswerable samples:

- `expected_doc` must be an empty string.
- `question_type` must be `unanswerable`.
- `ground_truth` must be `当前文档中未找到充分依据。`.
