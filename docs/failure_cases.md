# Failure Cases

## q006 - naive

- question: 这个文档有没有提到火星移民计划？
- expected_doc: 
- top citations: sample.txt, sample.md, sample.md
- failure_reason: fallback_error

## q012 - naive

- question: 这些文档有没有提到量子传送门？
- expected_doc: 
- top citations: CMD3_Cross-Modal_Decoupled_Deformable_Distillation_for_EEG-fNIRS_Fusion.pdf, sample.txt, CMD3_Cross-Modal_Decoupled_Deformable_Distillation_for_EEG-fNIRS_Fusion.pdf
- failure_reason: fallback_error

## q024 - naive

- question: 这些文档有没有描述区块链支付协议？
- expected_doc: 
- top citations: CMD3_Cross-Modal_Decoupled_Deformable_Distillation_for_EEG-fNIRS_Fusion.pdf, sample.md, sample.md
- failure_reason: fallback_error

## q006 - hybrid

- question: 这个文档有没有提到火星移民计划？
- expected_doc: 
- top citations: CMD3_Cross-Modal_Decoupled_Deformable_Distillation_for_EEG-fNIRS_Fusion.pdf, CMD3_Cross-Modal_Decoupled_Deformable_Distillation_for_EEG-fNIRS_Fusion.pdf, sample.txt
- failure_reason: fallback_error

## q012 - hybrid

- question: 这些文档有没有提到量子传送门？
- expected_doc: 
- top citations: CMD3_Cross-Modal_Decoupled_Deformable_Distillation_for_EEG-fNIRS_Fusion.pdf, CMD3_Cross-Modal_Decoupled_Deformable_Distillation_for_EEG-fNIRS_Fusion.pdf, sample.txt
- failure_reason: fallback_error
