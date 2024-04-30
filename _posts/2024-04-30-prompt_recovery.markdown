---
layout:       post
title:        "[LLM Prompt Recovery](PB:0.6695)银牌方案总结(1)"
author:       "Roschild.Rui"
header-style: text
catalog:      true

---
> “[比赛内容细节](https://www.kaggle.com/competitions/llm-prompt-recovery)”

我在接下来的内容中会先简单介绍我们团队的最终提交方案

随后会将整个方案的形成过程在接下来的blog中总结呈现

再次感谢开放，充满活力的[kaggle](https://www.kaggle.com/)数据科学社区竞赛平台

### 方案总结
- 1.基于deberta-v3-large的seq2seq模型
- 2.通过构建合适的adapter层微调[phi2](https://www.kaggle.com/models/Microsoft/phi/Transformers/2/1)模型
- 3.few-shot [mistral-7b-v2](https://www.kaggle.com/datasets/ahmadsaladin/mistral-7b-it-v02)模型

将三个模型还原的提示词集成在一起作为最终的预测结果
