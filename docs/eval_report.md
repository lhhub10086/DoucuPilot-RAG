# Evaluation Report

## Evaluation Setup

- documents: local `data/raw` corpus
- chunk_strategy: `section`
- eval_file: `eval_sets\sample_eval.jsonl`
- eval_samples: 25
- answerable_samples: 22
- unanswerable_samples: 3
- strategies: naive, hybrid, rerank, agentic
- embedding_model: `BAAI/bge-small-zh-v1.5`
- reranker_model: `BAAI/bge-reranker-base`

## Strategy Comparison

| strategy | hit_rate@5 | mrr | citation_accuracy | keyword_coverage | fallback_accuracy_on_unanswerable | avg_latency_ms |
|---|---:|---:|---:|---:|---:|---:|
| naive | 1.000 | 0.977 | 1.000 | 0.912 | 0.000 | 2069.2 |
| hybrid | 1.000 | 1.000 | 1.000 | 0.950 | 0.000 | 7.4 |
| rerank | 1.000 | 1.000 | 1.000 | 0.974 | 0.667 | 1257.9 |
| agentic | 1.000 | 1.000 | 1.000 | 0.974 | 1.000 | 789.3 |

## Unanswerable Performance

| strategy | unanswerable_total | fallback_count | forced_answer_count | fallback_accuracy_on_unanswerable |
|---|---:|---:|---:|---:|
| naive | 3 | 0 | 3 | 0.000 |
| hybrid | 3 | 0 | 3 | 0.000 |
| rerank | 3 | 2 | 1 | 0.667 |
| agentic | 3 | 3 | 0 | 1.000 |

## Observations

- hybrid hit_rate@5 vs naive: +0.000.
- rerank citation_accuracy vs hybrid: +0.000.
- agentic fallback_accuracy_on_unanswerable: 1.000.
- highest average latency: naive.

## Failure Cases

### q006 - naive

- question: 这个文档有没有提到火星移民计划？
- expected_doc: 
- top citations: sample.txt, sample.md, sample.md
- failure_reason: fallback_error

### q012 - naive

- question: 这些文档有没有提到量子传送门？
- expected_doc: 
- top citations: CMD3_Cross-Modal_Decoupled_Deformable_Distillation_for_EEG-fNIRS_Fusion.pdf, sample.txt, CMD3_Cross-Modal_Decoupled_Deformable_Distillation_for_EEG-fNIRS_Fusion.pdf
- failure_reason: fallback_error

### q024 - naive

- question: 这些文档有没有描述区块链支付协议？
- expected_doc: 
- top citations: CMD3_Cross-Modal_Decoupled_Deformable_Distillation_for_EEG-fNIRS_Fusion.pdf, sample.md, sample.md
- failure_reason: fallback_error

### q006 - hybrid

- question: 这个文档有没有提到火星移民计划？
- expected_doc: 
- top citations: CMD3_Cross-Modal_Decoupled_Deformable_Distillation_for_EEG-fNIRS_Fusion.pdf, CMD3_Cross-Modal_Decoupled_Deformable_Distillation_for_EEG-fNIRS_Fusion.pdf, sample.txt
- failure_reason: fallback_error

### q012 - hybrid

- question: 这些文档有没有提到量子传送门？
- expected_doc: 
- top citations: CMD3_Cross-Modal_Decoupled_Deformable_Distillation_for_EEG-fNIRS_Fusion.pdf, CMD3_Cross-Modal_Decoupled_Deformable_Distillation_for_EEG-fNIRS_Fusion.pdf, sample.txt
- failure_reason: fallback_error
