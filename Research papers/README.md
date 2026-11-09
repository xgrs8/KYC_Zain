# Research papers

These papers document the model and labeling choices for the KYC_Zain
workflow. The PDFs are included unchanged for reference. Both arXiv records
identify the papers as available under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## DocRes

**Paper:** Jiaxin Zhang, Dezhi Peng, Chongyu Liu, Peirong Zhang, and Lianwen Jin,
"DocRes: A Generalist Model Toward Unifying Document Image Restoration Tasks"
(2024), [arXiv:2405.04408](https://arxiv.org/abs/2405.04408).

**Brief:** DocRes presents one generalist document-restoration model for
dewarping, deshadowing, appearance enhancement, deblurring, and binarization,
guided by task-specific visual prompts.

**Project use:** DocRes is the neural network used for document-image
enhancement in place of Qwen Image 2.1. Its restoration tasks target common
document-photo problems such as blur, shadows, distortion, and uneven
appearance. The paper also reports a failure case for end-to-end enhancement
of camera-captured documents, so outputs should be checked against the source
image.

**Repository scope:** This repository contains the paper, but does not include
DocRes inference code or model weights.

PDF: [`2405.04408v1.pdf`](2405.04408v1.pdf)

## CALICO

**Paper:** Boqin Yuan, Xiaoyi Gu, Fiona Li, Chang Wan, Angel Hsing-Chi Hwang,
and Jieyu Zhao, "CALICO: A Human-Centered, Codebook-Aligned System for
Annotation" (2026), [arXiv:2609.14726](https://arxiv.org/abs/2609.14726).

**Brief:** CALICO describes a human-centered annotation workflow that turns
codebooks into inspectable prompts and labels. It supports prompt generation,
versioning, human feedback, review, and label-supervised prompt optimization.
The paper evaluates it on codebooks for AI-companion conversations.

**Project use:** CALICO is used for professional, codebook-aligned labeling:
clear decision rules, reviewable outputs, and a process for improving
annotation instructions with human feedback. Its published evaluation is in a
different annotation domain, so the KYC document workflow still needs its own
document-specific schema and quality checks.

**Repository scope:** This repository contains the paper, not the CALICO web
application. The A.L.G software and its MCP server are the KYC_Zain labeling
tools included here.

PDF: [`2609.14726v1.pdf`](2609.14726v1.pdf)
