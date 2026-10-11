---
title: Full Profile
---

I focus on multi-turn interactive learning for LLM agents and long-term stable decision-making in multi-agent systems, with a sustained interest in long-horizon reward rubrics, credit assignment, verifiable supervision signals, and stable training mechanisms. I can drive the full loop from research hypotheses to reproducible experiments and on to online or system-level validation, and I'm good at quickly locating bottlenecks in complex settings, proposing solutions, and pushing them through to deployment.

---

### Education

- **(2027 - 2030, expected) The University of Hong Kong** · Ph.D., advised by [Qi Liu](https://leuchine.github.io/)
- **(2023 - 2027) South China University of Technology** · B.Eng. in Data Science and Big Data · GPA 3.79/4.00
  - 2023-2024 Comprehensive Evaluation: ranked 1/89 in major; School-level First-Class Scholarship; School-level Excellent Student
  - First Prize, Sci-Tech Innovation Award, Future Technology Taihu Innovation Award, 2024
- **(Jul 2024 - Aug 2024) National University of Singapore** · Summer School in Artificial Intelligence and Machine Learning · Distinction

---

### Experience

- **(Jun 2026 - Present) Grace Investment Machine · AI Quantitative Researcher (Remote)**
  - Building an autonomous research and self-evolving agent system for live trading in the A-share market. It continuously discovers factors and features, generates and validates financial hypotheses, iterates predictive models, and searches for effective strategy combinations through an automated research-to-trading loop. Strategies developed autonomously by the system are running in live trading.
- **(Jul 2025 - Aug 2026) Tencent · Agent Research - NLP (Qingyun Project)**
  - Built a deep search data synthesis pipeline for the financial domain, featuring a Hard-to-Infer, Easy-to-Verify multi-turn, multi-level tool-calling framework.
  - Led the multi-turn financial deep search agent training project (256+ GPUs). With an improved credit assignment strategy and multi-level reward rubrics, models trained on synthesized data at different levels improved by 15% and 24% on the Greater China T2 / T3 sets of FinSearchComp. The model is integrated into Tencent Yuanbao and ranked #2 globally on FinSearchComp Greater China (as of 2025-09).
  - Core member of the inference-time sampling acceleration for the MTP module of a multi-turn tool-use agent model (1.4× speedup). Also drove acceleration of a general reward model for rubric-based evaluation of multi-step tool use, enabling base models such as GPT-OSS-120B with SpecForge-trained EAGLE3 to reach up to 2.7× speedup.
- **(TBD) ByteDance · Jindouyun Talent Program · LLM Algorithm Intern**

---

### Publications

- **ACL 2026 Findings** (first author) · [arXiv](https://arxiv.org/abs/2605.14752)<br>Cognitive-Uncertainty Guided Knowledge Distillation for Accurate Classification of Student Misconceptions<br>**Qirui Liu**, Hao Chen, Weijie Shi, Jiajie Xu, Jia Zhu
- **NeurIPS 2026 (under review)**<br>Grounded Credit Assignment in Reinforcement Learning for Multi-Turn Search Agents<br>**Qirui Liu**, Hao Chen, Weijie Shi, Xing Tang, Weihong Luo, Xiku Du, Yuyu Luo
- **KDD 2026 ADS** · [arXiv](https://arxiv.org/abs/2605.31064)<br>Fighting Numerical Hallucinations via Data-centric Compilation for Online Financial QA<br>Hao Chen, Xing Tang, **Qirui Liu**, Weijie Shi, Shiwei Li, Fuyuan Lyu, Weihong Luo, Xiku Du, Xiuqiang He
- **EMNLP 2026 Findings**<br>FinHopper: A Progressive Multi-Hop Benchmark for Complex Financial Agents with Step-wise Evaluation<br>Yuxuan Wu, Miaopei Lin, Qiufang Ying, Lin Dan, Hao Zhao, **Qirui Liu**, Yuteng Shen, Hao Chen, Shaohui Wu, Weihong Luo, Xiku Du

---

### Competitions

- **(Jun 2026) [NVIDIA Nemotron Model Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge)** · 87/4182 · Silver<br>Claude Opus 4.6 + 64×H20 + a custom agent swarm semi-automatically built LoRA SFT / GSPO / OPD training frameworks (ms-swift, verl; FSDP / Megatron) for the niche Mamba model, and fully automatically found a fairly reliable offline test set
- **(Jun 2026) [BirdCLEF+ 2026](https://www.kaggle.com/competitions/birdclef-2026)** · 96/4094 · Silver<br>Claude Opus 4.6 + a custom agent swarm ran fully autonomously and finished in about a month
- **(Oct 2025) [Jigsaw - Agile Community Rules Classification](https://www.kaggle.com/competitions/jigsaw-agile-community-rules)** · 95/2445 · Silver
- **(Oct 2025) [MAP - Charting Student Math Misunderstandings](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings)** · 59/1858 · Silver
- **(Jul 2025) [KDD Cup 2025 CRAG-MM Challenge](https://www.aicrowd.com/challenges/meta-crag-mm-challenge-2025)** · 9th overall, 5th in Track 1 (multimodal agents; single-hop / multi-hop / multi-turn RAG)
- **(Mar 2025) [WSDM Cup - Multilingual Chatbot Arena](https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena)** · 15/950 · Silver
- **(Dec 2024) [Eedi - Mining Misconceptions in Mathematics](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics)** · 19/1446 · Silver
- **(Aug 2024) [LMSYS - Chatbot Arena Human Preference Predictions](https://www.kaggle.com/competitions/lmsys-chatbot-arena)** · 19/1849 · Silver
- **(Jun 2024) [AI Mathematical Olympiad - Progress Prize 1](https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize)** · 58/1161 · Silver
- **(May 2024) [Home Credit - Credit Risk Model Stability](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability)** · 350/2175 · Bronze
- **(Apr 2024) [LLM Prompt Recovery](https://www.kaggle.com/competitions/llm-prompt-recovery)** · 25/2175 · Silver · [solution write-up](/2024/04/30/prompt_recovery/)

Others: Second Prize, 2023 Asia-Pacific Mathematical Contest in Modeling; Meritorious Winner, 2024 Huashu Cup International Mathematical Modeling Contest; Second Prize, 2024 University Big Data Challenge; Honorable Mention, 2024 MCM/ICM; Regional Third Prize, 2024 MotherCup; Meritorious Winner, 2025 MCM/ICM

---

### Quantitative Trading (personal, live results)

- **Multi-factor + agent research** (2026-05-21 to 2026-06-26): a few simple delay-1 factors plus a Claude Code–based multi-agent automated research swarm returned 81.35%, with a 6.8% max drawdown and 64.22% excess return over the STAR 50 Index
- **Small-cap strategy** (2024-10-22 to 2025-08-15): 62.43% cumulative return, 82.28% annualized, 20.94% max drawdown, Sharpe 2.693
- **Limit-up breakout strategy** (2025-01-01 to 2026-01-01): 269.19% cumulative return, 283.34% annualized, 25.44% max drawdown, Sharpe 4.777

---

> Updated August 31, 2026
