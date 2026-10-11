---
date: 2026-10-09 21:30:00 +0800
en: |
  Yesterday I read an article analyzing issues with DSV4’s CSA attention mechanism: <https://arxiv.org/pdf/2609.36322>

  I revisited DeepSeek’s foundation models and came away with the impression that many of its architectural innovations—whether MLA, CSA, or MHC—are essentially derived from quantitative combo strategy models.

  DSV4.1 Flash technical report: <https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf>  
  DSV4 technical report: <https://arxiv.org/pdf/2606.19348>

  I asked Opus 5.5 Max to write an HTML report based on the prompt below and the combo experiment data: [link](/2026/10/11/combo2llmmodel/en/). Sharing it here for reference.

  ```markdown
  I believe many of the tricks and insights from current combo research are also relevant to the development of LLM foundation models. Examples include the new architectural innovations introduced in the following two technical reports:

  DSV4.1 Flash technical report:
  https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf

  DSV4 technical report:
  https://arxiv.org/pdf/2606.19348

  Conduct in-depth research and analysis on the following:

  1. The connections between existing combo tricks and insights and the new architectures introduced in these two technical reports.
  2. New tricks and insights from existing combo research that may be effective for future foundation-model development.
  3. How existing combo research could inspire research on loop-transformer foundation models.
  4. Tricks and insights from these two new models that could help further improve current combo approaches.

  Write the findings as an HTML report.
  ```
---
昨天看了一篇文章 dsv4 csa注意力机制问题分析<https://arxiv.org/pdf/2609.36322>
重新回顾了一下ds的这些基模 感觉ds很多的基模架构上的创新 不管是mla csa mhc 基本都是基于量化combo策略模型来衍生的
dsv4.1 flash 技术报告<https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf>
dsv4 技术报告 <https://arxiv.org/pdf/2606.19348>
我让opus-5.5 max基于下面prompt和combo的实验数据撰写了一篇html [链接](/2026/10/11/combo2llmmodel/)供参考 
```markdown
我觉得现在很多combo研究的trick和insight在llm基础模型上的研发上都有所涉及 比如dsv4.1 flash 技术报告https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf
dsv4 技术报告 https://arxiv.org/pdf/2606.19348两个技术报告中创新的新架构 
深度研究并分析以下内容
1. 现有combo trick、insight和这两个模型技术报告新架构的关联
2. 现有combo对未来基础模型研发可能有效的新trick和insight
3. 现有combo对loop transformer基础模型有关研究启发
4. 上述两个新模型对现在combo后续提升的trick、insight
撰写html报告
```
