---
layout:       post
title:        "[LLM Prompt Recovery](PV:0.6694)银牌方案总结(1)"
author:       "Roschild.Rui"
header-style: text
catalog:      true

---
> “这是我的第一个blog，希望在这一路上能认识更多志同道合的朋友”

> "This is my first blog post, marking the start of my exciting journey to become a Kaggle Grandmaster. Setting out on this path, I look forward to meeting with more friends who are equally passionate about data science. Together, we will dive into the fascinating world of machine learning, uncover valuable insights, and strive to make a meaningful impact in the field."

### [比赛内容](https://www.kaggle.com/competitions/llm-prompt-recovery)
我首先在这一篇blog中会先简单介绍我们团队的最终提交方案

随后整个方案的迭代形成过程将在接下来我的blog中逐一总结呈现

同时，我们也会在接下来的blog中通过复现大佬的比赛方案，反思总结，以期实现更好成果产出

感谢 [Kaggle](https://www.kaggle.com/) ——一个开放，富有活力的数据科学社区竞赛平台，为我们提供了一次深入理解LLM内部机制的机会。

### 方案总结
- 1.构建基于deberta-v3-large的seq2seq模型
- 2.构建合适的adapter层微调[phi2](https://www.kaggle.com/models/Microsoft/phi/Transformers/2/1)模型
- 3.few-shot [mistral-7b-v2](https://www.kaggle.com/datasets/ahmadsaladin/mistral-7b-it-v02)模型

将三个模型还原的提示词集成在一起作为最终的预测结果

### 数据

**reference**
- https://www.kaggle.com/code/tomooinubushi/all-in-one-dataset-with-embedding/notebook
- https://huggingface.co/datasets/Skylion007/openwebtext
- https://huggingface.co/datasets/euclaise/writingprompts/viewer/default/train?p=1
