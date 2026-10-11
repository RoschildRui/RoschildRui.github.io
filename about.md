---
layout: page
title: 完整介绍
permalink: /about/
---

关注 LLM agent 的多轮交互学习与多 agent 系统中的长期稳定决策，持续关注长程 reward rubric、credit assignment、可验证的监督信号与稳定的训练机制。能推动从研究假设到可复现实验、再到线上 / 系统级验证的完整闭环；擅长在复杂场景中快速定位瓶颈、提出方案并推动落地。

---

### 教育经历

- **（2027 - 2030，预期）香港大学** · 博士，导师 [刘琦](https://leuchine.github.io/)
- **（2023 - 2027）华南理工大学** · 数据科学与大数据 本科 · GPA 3.79/4.00
  - 2023-2024 学年综合素质评价 专业 1/89，校级一等奖学金、校级优秀学生
  - 2024 年度未来技术太湖创新奖 · 科创创新奖一等奖
- **（2024.7 - 2024.8）新加坡国立大学** · 人工智能与机器学习暑期学校 · Distinction

---

### 实习经历

- **（2026.6 - 至今）Grace Investment Machine · AI 量化研究员（远程）**
  - 搭建面向 A 股实盘交易的自主研究、自进化 agent 系统：持续挖掘因子与特征、生成并验证金融假设、迭代预测模型，在自动化的「研究—交易」闭环中搜索有效的策略组合；系统自主产出的策略已部署实盘。
- **（2025.7 - 2026.8）腾讯 · Agent 研究 - NLP（青云计划）**
  - 负责金融领域 deep search 数据合成管线，构建「难推断、易验证」的多轮、多层级工具调用框架。
  - 主导多轮金融 deep search agent 模型训练项目（256+ GPU）。通过改进的 credit assignment 策略与多层级 reward rubric 设计，不同层级合成数据训练的模型在 FinSearchComp 大中华区 T2 / T3 上分别提升 15% 与 24%；模型已接入腾讯元宝，在 FinSearchComp 大中华区数据表现上全球第 2（截至 2025-09）。
  - 多轮工具调用 agent 模型 MTP 模块推理采样加速的核心成员，提速 1.4×；推动面向多步工具调用 rubric 评测的通用奖励模型加速，GPT-OSS-120B 等基座配合 SpecForge 训练的 EAGLE3 最高提速 2.7×。
- **（待讨论）字节跳动 · 筋斗云人才计划 · 大模型算法实习生**

---

### 论文

- **ACL 2026 Findings**（第一作者）· [arXiv](https://arxiv.org/abs/2605.14752)<br>Cognitive-Uncertainty Guided Knowledge Distillation for Accurate Classification of Student Misconceptions<br>**Qirui Liu**, Hao Chen, Weijie Shi, Jiajie Xu, Jia Zhu
- **NeurIPS 2026（审稿中）**<br>Grounded Credit Assignment in Reinforcement Learning for Multi-Turn Search Agents<br>**Qirui Liu**, Hao Chen, Weijie Shi, Xing Tang, Weihong Luo, Xiku Du, Yuyu Luo
- **KDD 2026 ADS** · [arXiv](https://arxiv.org/abs/2605.31064)<br>Fighting Numerical Hallucinations via Data-centric Compilation for Online Financial QA<br>Hao Chen, Xing Tang, **Qirui Liu**, Weijie Shi, Shiwei Li, Fuyuan Lyu, Weihong Luo, Xiku Du, Xiuqiang He
- **EMNLP 2026 Findings**<br>FinHopper: A Progressive Multi-Hop Benchmark for Complex Financial Agents with Step-wise Evaluation<br>Yuxuan Wu, Miaopei Lin, Qiufang Ying, Lin Dan, Hao Zhao, **Qirui Liu**, Yuteng Shen, Hao Chen, Shaohui Wu, Weihong Luo, Xiku Du

---

### 竞赛

- **（2026.6）[NVIDIA Nemotron Model Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge)** · 87/4182 · 银牌<br>Claude Opus 4.6 + 64×H20 + 自定义 agent swarm，半自动搭建了面向小众 Mamba 模型的 LoRA SFT / GSPO / OPD 训练框架（ms-swift、verl，FSDP / Megatron），并全自动找到了相对可靠的线下测试集
- **（2026.6）[BirdCLEF+ 2026](https://www.kaggle.com/competitions/birdclef-2026)** · 96/4094 · 银牌<br>Claude Opus 4.6 + 自定义 agent swarm 全自主运行，约一个月完成
- **（2025.10）[Jigsaw - Agile Community Rules Classification](https://www.kaggle.com/competitions/jigsaw-agile-community-rules)** · 95/2445 · 银牌
- **（2025.10）[MAP - Charting Student Math Misunderstandings](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings)** · 59/1858 · 银牌
- **（2025.7）[KDD Cup 2025 CRAG-MM Challenge](https://www.aicrowd.com/challenges/meta-crag-mm-challenge-2025)** · 总榜第 9，赛道一第 5（多模态 agent，单跳 / 多跳 / 多轮 RAG）
- **（2025.3）[WSDM Cup - Multilingual Chatbot Arena](https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena)** · 15/950 · 银牌
- **（2024.12）[Eedi - Mining Misconceptions in Mathematics](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics)** · 19/1446 · 银牌
- **（2024.8）[LMSYS - Chatbot Arena Human Preference Predictions](https://www.kaggle.com/competitions/lmsys-chatbot-arena)** · 19/1849 · 银牌
- **（2024.6）[AI Mathematical Olympiad - Progress Prize 1](https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize)** · 58/1161 · 银牌
- **（2024.5）[Home Credit - Credit Risk Model Stability](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability)** · 350/2175 · 铜牌
- **（2024.4）[LLM Prompt Recovery](https://www.kaggle.com/competitions/llm-prompt-recovery)** · 25/2175 · 银牌 · [方案总结](/2024/04/30/prompt_recovery/)

其他：2023 亚太杯二等奖、2024 华数杯国际数学建模大赛 M 奖、2024 高校大数据挑战赛二等奖、2024 美赛 H 奖、2024 MotherCup 赛区三等奖、2025 美赛 M 奖

---

### 量化交易（个人，实盘数据）

- **多因子 + agent 投研**（2026.5.21 - 2026.6.26）：几个简单的 delay-1 因子 + 基于 Claude Code 的多 agent 自动投研 swarm，收益 81.35%，最大回撤 6.8%，相对科创 50 超额收益 64.22%
- **小市值策略**（2024.10.22 - 2025.8.15）：累计收益 62.43%，年化 82.28%，最大回撤 20.94%，夏普 2.693
- **涨停突破策略**（2025.1.1 - 2026.1.1）：累计收益 269.19%，年化 283.34%，最大回撤 25.44%，夏普 4.777

---

> 更新于 2026 年 8 月 31日
