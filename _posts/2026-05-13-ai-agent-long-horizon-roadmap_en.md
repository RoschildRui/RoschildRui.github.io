---
title: "Bottlenecks & an Algorithmic Roadmap for Long-Horizon AI Agents"
---

> Today's strongest model (claude-opus-4-7) paired with the strongest agent harness (claude code) still struggles to match a professional human researcher on long-horizon, complex tasks such as investment research. In this post we lay out the two core problems with current systems, and propose an algorithmic roadmap that goes from "getting it right" to "getting it fast."
>
> The full technical detail is preserved, while key terms are explained in plain language. A **Glossary** is included at the end for readers without an agent / ML background.

## What Counts as a "Long-Horizon Task"

By "long-horizon, complex task" we mean the kind of task that requires an agent to work continuously for a long time, call many tools, and span many steps before it's done. Two typical examples:

- **Deep-diving a sector or a single company** — a one-line request like "research electronic-grade glass cloth for me" sits on top of hundreds of related companies, the workforces behind them, the supply chains and the macro/micro economic links to countries and the wider world, plus countless research notes, financial filings, and earnings-call transcripts to gather, retrieve, cross-validate, and synthesize.
- **Solving an algorithm problem that would take an experienced researcher + engineer 1–2 months.**

Ideally you'd hand such a task to "the strongest model (e.g. the opus family) + the strongest harness (e.g. the latest Claude Code) + a slash command," and it would coordinate a swarm of sub-agents to finish the job on its own. In reality, it often does this poorly. Where does it break down?

## Part 1 — Two Core Problems with Current Systems

### Problem 1: Skill interfaces are too rigid — no collaboration, no multi-source validation, and the data is wrong

Many skills (the "abilities/tools" an agent can call) have interfaces written far too rigidly — they neither let multiple skills or agents collaborate, nor cross-check across multiple sources. The result is badly inaccurate data, and in investment research, wrong data means wrong conclusions.

Two real potholes we've hit:

- **Example 1 (stale valuation data):** A skill outputs "analysts forecast 2026 net profit attributable to the parent of ¥6.0bn (+83% YoY), implying a PE of ~16x." The problem: if you really run the numbers on ¥6.0bn of earnings, the PE at the current share price is no longer 16x. That 16x was computed off the **March share price** — the data is stale.
- **Example 2 (stale price data):** A skill reports "the price of 7628 electronic cloth rose from ¥4.15/m at end-Q3 2025 to about ¥5.8/m currently, up **+40%**." But that so-called "current price" is, again, the **March price**.

The shared root cause: as context grows, the agent in a long-horizon task tends to slip on validating what a skill returns. The classic symptom is grabbing a single number from a skill interface and calling it done — never tagging its as-of date, never reconciling it against another source.

**The shape of the fix is already visible.** As shown below, spinning up a fresh Claude Code session with the neodata skill retrieves the **latest** price of "electronic cloth" as quoted in research notes.

![Querying the latest electronic-cloth price from research notes via Claude Code + the neodata API](/img/in-post/electronic-cloth-price-demo.png)

### Problem 2: In long-horizon tasks the agent team "degrades," and multimodal information is handled poorly

This one is really two sub-problems.

**(a) The agent team collectively "degrades" as context accumulates.**
At the start of a task, the main agent can still coordinate sub-agents / agent teams normally. But as context piles up — **even when you've built a multi-level harness (coordinator → planner → executor)** — sub-agents / agent teams get called less and less often, and the capability of every layer (high, middle, execution) declines together. By the end the whole system seems to be "rushing to hand something in": task completion is low and falls short of the goal (the completion bar) set at the outset.

**(b) Multimodal information — audio especially — is handled both poorly and unfairly.**
Current ASR models (which transcribe speech to text) can't handle multimodal information well: they're bad at analyzing the **tone, attitude, and emotion** of a speaker on an earnings call. More importantly there's an *unfairness* — when weighed against same-tier information (financial filings, institutional research notes), earnings-call content is clearly not treated as an equal: the coordinator / planner agents tend to treat this audio information as **noise**. The consequence is that, in a long-horizon task, the system often calls ASR **once** to get the earnings-call content and then **never calls it again** — throwing away an important source for free.

## Part 2 — The Algorithmic Roadmap

The roadmap splits into two halves: one for "getting the long-horizon task **right**" (how to manage the agent swarm), and one for "getting it **fast**" (accelerating long, multi-turn tool calling).

### (A) Agent Swarm Management: chewing through long-horizon tasks with a multi-level harness

#### 1. An atomized, multi-level agent architecture

The core idea is one sentence: **except for a single main agent, every other agent (coordinator, planner, executor) is "atomized"** — spun up and torn down on demand, discarded after use, so that each agent's context stays clean. This attacks the "context-accumulation degradation" of Problem 2(a) at the root. Every type and every group of agents can access skills / MCP / tools.

The division of labor looks like this:

- **Main agent:** discusses and clarifies the full task with the user, and fixes the **completion bar (delivery criteria)**; takes the coordinator agents' results, updates files / folders, and passes them on to the next coordinator group. It also serves as the later **verifier** — using the criteria agreed with the user up front to judge whether the coordinator group actually finished the task.
- **Coordinator agent group:** discusses and coordinates the full task and completion bar handed down by the main agent, breaks it into **exploratory** sub-tasks, and assigns them to the planner agent group.
- **Planner agent group (+ exploratory-planning skill + skill router):** plans "how to explore and how to finish" this task, draws a complete exploration / completion **path map**, and hands the deterministic parts to the executor agent group; then, based on what comes back, keeps refining the path map → dispatches the next executor group, or returns results to the coordinator agent.
- **Executor agent group (+ deterministic-execution skill + skill router):** does only **deterministic** work and returns **deterministic** results.

In one line: **the main agent answers to the user (define the task + accept the result), the coordinator agent splits exploratory tasks, the planner agent does the exploratory planning, and the executor agent does the deterministic execution.** Exploration (uncertain) and execution (certain) are cleanly separated across layers.

#### 2. Self-evolved agent skills

Skills shouldn't be hard-coded. The idea: start from a first version of a skill written by hand / via "vibe coding," then iterate it against the agent's real execution environment so it gets better the more it's used.

There are **two directions** of evolution:

- **A. Evolve along the data interface** — the goal is a more efficient and more accurate retrieval / execution path (which directly addresses the "rigid interface, wrong data" of Problem 1).
- **B. Evolve along user preference** — the goal is output whose style and format increasingly match what the user wants.

And on *how* to evolve, there are two opposite approaches:

- **Approach 1:** make a single skill as exhaustive as possible, covering every case (big-and-complete).
- **Approach 2:** split a skill into finer sub-skills, with high-level skills calling base skills (small-and-composable).

So how do you judge whether a skill is evolving in the right direction? The answer — a **rubric (scoring standard).**

#### 3. The dynamic rubric model

A task's complete scoring standard has an **objective** and a **subjective** half:

- **A — objective standard:** the task's own quantifiable completion criteria, turned into objective rubrics.
- **B — subjective standard:** the completion bar from the user's point of view. The method is to **simulate different personas** — professional analyst / researcher, power user, novice — and build a subjective rubric for each.

Put A + B together and you get a complete scoring standard for a problem, which you then use to judge "whether the skill is evolving in the right direction." It's called *dynamic* because this rubric shifts with the task and with the persona.

#### 4. Skill Router (recall + rerank)

Most current harnesses decide which skill to call by **looking only at the skill's name and description**. But once you have a large number of skills whose names / descriptions heavily overlap, name-and-blurb alone can't pick the right one.

The solution: **decouple "skill selection" out of the harness**, and separately train two small models (~0.6B each) into a **recall + rerank** skill router (recall candidates first, then fine-rank). The input to both models is every skill's **name + description + body (i.e. the skill's source / implementation)**, scored pointwise / listwise.

Why feed in the body too? The report cites the paper **SkillRouter: Skill Routing for LLM Agents at Scale** (arXiv:2603.22455), which argues that **the body of a skill is necessary for picking the right skill** — name and description aren't enough.

#### 5. ASR model + HuBERT

For Problem 2(b) — earnings-call audio treated as noise, with tone and emotion missed — the roadmap's direction is to bring in an **ASR model + HuBERT** (a self-supervised speech-representation model) to handle earnings-call audio / multimodal information more solidly, so it gets a fair weight alongside filings and research notes.

### (B) Accelerating long, multi-turn tool calling

Once the task is done right, it still has to be done fast. A long-horizon task easily runs tens to hundreds of tool calls, and speed is a hard constraint.

#### 1. Model-level acceleration

The idea is to **treat each agent role differently**, because the work each role does is fundamentally different in nature:

- **Executor agent (deterministic-task executor):** for example, the planner agent hands it 3 PDF financial filings and asks it to compute some metric, or to judge whether a conclusion holds — a process that can be fully governed by a specific skill / SOP (standard operating procedure). These tasks are **extremely accuracy-sensitive but highly programmable**. So the acceleration recipe is: **programmatically synthesized trajectories** (building on prior practice + a specific skill / problem set) **+ small-model SFT** (around 4B / A3B) **+ speculative sampling**.
- **Planner agent:** speculative sampling (mtp / eagle3 / dflash, etc.).
- **Coordinator agent:** likewise speculative sampling (mtp / eagle3 / dflash).

Why single out the planner and coordinator layers? Because they handle **open-ended, exploratory tasks with large outcome uncertainty**, so the base model should be a large open-source SOTA model — **glm5.1, dsv4 pro, hy3 preview** and the like — which we then accelerate for our own tasks. The priority order of acceleration approaches is:

1. **speculative sampling** — speed up the original model's inference on a specific task (without changing the model's own capability);
2. **distillation** — collect enough trajectories / logprobs from SOTA models (closed- or open-source) and distill (sft / opd / rl) them into our own small model.

#### 2. Evaluating completion quality after acceleration

The big risk of acceleration is "faster but dumber." So the final step is to use the same **dynamic rubric model** from above to evaluate task-completion quality *after* acceleration, making sure the speedup didn't sacrifice quality. Judging "skill-evolution direction" and accepting "post-acceleration quality" use the same rubric system — closing the loop.

## Closing

Stringing it all together, the logic is actually clear:

- **Get it right:** atomized multi-level agents (clean context) + self-evolved skills (more accurate the more they're used) + dynamic rubrics (spelling out what "done well" means) + a skill router (picking the right tool out of a sea of skills) + ASR/HuBERT (no longer dropping earnings-call information);
- **Get it fast:** accelerate per role — the deterministic execution layer goes "synthesized trajectories + small-model SFT + speculative sampling," and the exploratory planning / coordination layer goes "large-model base + speculative sampling / distillation";
- **Close the loop:** use one and the same dynamic rubric to both judge skill-evolution direction and accept post-acceleration quality.

There's only one end goal: to make AI agents genuinely capable of the long-horizon, complex investment-research tasks that **used to take an expert 1–2 months** — with fresh data, complete sources, a fast process, and correct conclusions.

---

## Glossary

| Term | One-line explanation |
| --- | --- |
| **harness** | The agent's "shell / orchestration framework" — it governs how the model calls tools and coordinates multiple agents (e.g. Claude Code). |
| **skill** | One "ability / tool" an agent can call; usually has a name, a description, and a body (the implementation). |
| **MCP** | A protocol/interface that lets a model connect to external tools / data sources in a unified way. |
| **subagent / agent team** | Child agents or small agent groups spun off by the main agent, each handling its own slice. |
| **context** | Everything the model is currently "holding"; the more it piles up, the slower and lower-quality it gets. |
| **atomized agent** | An agent spun up and torn down on demand, discarded after use, to keep context clean. |
| **rubric** | A scoring rubric that breaks "how well it was done" into scorable criteria. |
| **recall + rerank** | The two-stage retrieval pattern: first "recall" a candidate set, then "fine-rank" the best — used here to pick a skill. |
| **ASR** | Automatic Speech Recognition — turning audio (e.g. an earnings call) into text. |
| **HuBERT** | A self-supervised speech-representation model, good at extracting features from audio (including tone / emotion cues). |
| **SFT** | Supervised Fine-Tuning — training a model further on labeled data. |
| **distillation** | Having a small model learn a large model's behavior / outputs, so it's small and fast without losing much quality. |
| **speculative sampling** | An inference-acceleration technique: a small model "guesses" several tokens, a large model quickly verifies them, and overall output is faster (mtp / eagle3 / dflash are specific schemes). |
| **logprob** | The log-probability a model assigns to each candidate token; often used in distillation / analysis. |
| **PE** | Price-to-Earnings ratio — share price ÷ earnings per share, a common valuation metric. |
| **net profit attributable to parent** | Net profit attributable to the shareholders of the parent company. |
| **7628 electronic cloth** | A grade of electronic-grade glass-fiber cloth; an upstream material for PCBs / copper-clad laminates. |
| **neodata / opus / glm5.1 / dsv4 pro / hy3 preview, etc.** | Specific data-interface / large-model codenames mentioned in the report (some are internal or third-party names). |

---

*This post was reorganized and rewritten by claude code opus 4.8 from "260513 shared report.docx," preserving all technical points of the original report and explaining the terminology in plain language.*
