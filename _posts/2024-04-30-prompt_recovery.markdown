---
layout:       post
title:        "[LLM Prompt Recovery](PV:0.6694)银牌方案总结(1)"
author:       "Roschild.Rui"
header-style: text
catalog:      true

---
> “这是我的第一个blog，希望在这一路上能认识更多志同道合的朋友”

> "This is my first blog post. Setting out on this path, I am looking forward to meeting more friends who also have greate passion on data science. Together, we will dive into the fascinating world of data, uncover valuable and deeper insights on this Earth, and strive to make a meaningful impact to make people life great 'again' !!!"

### 写在前面
在这一篇blog中我会先简单介绍我们团队的最终提交方案

随后整个方案的迭代形成过程将在接下来我的blog中逐一呈现

同时，我们也会在接下来的blog中通过复现金牌区以及部分优秀银牌区的比赛方案，反思总结，以期加深我们对于大模型的理解

感谢 [Kaggle](https://www.kaggle.com/) ——一个开放，富有活力的数据科学社区竞赛平台，为我们提供了一次深入理解LLM内部机制的机会。

### [比赛内容](https://www.kaggle.com/competitions/llm-prompt-recovery)
**Overview**

LLMs are commonly used to rewrite or make stylistic changes to text. The goal of this competition is to recover the LLM prompt that was used to transform a given text.

**Description**

NLP workflows increasingly involve rewriting text, but there's still a lot to learn about how to **prompt LLMs effectively**. This machine learning competition is designed to be a novel way to dig deeper into this problem.

The challenge: recover the LLM prompt used to rewrite a given text. You’ll be tested against a dataset of 1300+ original texts, each paired with a rewritten version from Gemma, Google’s new family of open models.

**Evaluation**

Evaluation Metric

For each row in the submission and corresponding ground truth, sentence-t5-base is used to calculate corresponding embedding vectors. The score for each predicted / expected pair is calculated using the Sharpened Cosine Similarity, using an exponent of 3. The SCS is used to attenuate the generous score given by embedding vectors for incorrect answers. Do not leave any rewrite_prompt blank as null answers will throw an error.

Submission File

The submission file should contain a header and have the following format:

```csv
id,rewrite_prompt
000aaa,"Rewrite this essay but do it using the writing style of Dr. Seuss"
111bbb,"Rewrite this essay but do it using the writing style of William Shakespeare"
222ccc,"Rewrite this essay but do it using the writing style of Tupac Shakur"
...
```


### 方案总结
- 1.构建基于deberta-v3-large的seq2seq模型
- 2.构建合适的adapter层微调[phi2](https://www.kaggle.com/models/Microsoft/phi/Transformers/2/1)模型
- 3.few-shot [mistral-7b-v2](https://www.kaggle.com/datasets/ahmadsaladin/mistral-7b-it-v02)模型

将三个模型还原的提示词集成在一起作为最终的预测结果

### 数据集的构建

因为比赛方提供的数据**有限甚至等同于没有**，训练集和测试集都只有**一条**，所以我们需要额外查找生成一些数据以方便训练**端到端模型**以及**构建本地cv库**

#### 数据来源
以下是我们的数据来源，这里必须要感谢kaggle**开源数据集的大佬们**，以及**整理开源数据的开源大佬们**！！！

**reference**
- [https://www.kaggle.com/code/tomooinubushi/all-in-one-dataset-with-embedding/notebook](https://huggingface.co/datasets/Skylion007/openwebtext)
- [https://huggingface.co/datasets/Skylion007/openwebtext](https://huggingface.co/datasets/Skylion007/openwebtext)
- [https://huggingface.co/datasets/euclaise/writingprompts/viewer/default/train?p=1](https://huggingface.co/datasets/euclaise/writingprompts/viewer/default/train?p=1)
- [https://www.kaggle.com/datasets/nbroad/gemma-rewrite-nbroad](https://www.kaggle.com/datasets/nbroad/gemma-rewrite-nbroad)
- [https://www.kaggle.com/datasets/ilanmeissonnier/chatgpt-rewrite-promts/data](https://www.kaggle.com/datasets/ilanmeissonnier/chatgpt-rewrite-promts/data)
- [https://www.kaggle.com/datasets/winddude/70k-prompt-rewrite-triples/data](https://www.kaggle.com/datasets/winddude/70k-prompt-rewrite-triples/data)
- [https://www.kaggle.com/code/aatiffraz/generating-data-through-gemma7b-1000-texts-5h](https://www.kaggle.com/code/aatiffraz/generating-data-through-gemma7b-1000-texts-5h)
- ...

#### 数据预处理

```python
import re

def fix_prompt(text):
    patterns = [
        r'therefore.*I cannot',
        "does not contain any",
        'am unable to provide',
        "am unable to rewrite",
        "do not have the capacity to write",
        "am unable to engage",
        "I am unable to",
        "not provide information",
    ]
    sub_patterns= [
        r"Sure, here.*?:",
        r"Sure. here.*?:",
        r"Certainly, here.*?:",
        r"here.*? text*?:",
        r"Summary of .*?\n\n",
        r"Analysis of .*?\n\n",
        #r"The text.*?:"
    ]
    for p in patterns:
        match = re.search(p, text, re.IGNORECASE)
        
        if match:
            text = ''
    
    if text=="":
        return text
    else:
        for p in sub_patterns:
            match = re.search(p, text, re.IGNORECASE)
            if match:
                text = text[match.end():]
        if text.startswith('**\n\n') or text.startswith('user'):
            text=text[4:].strip()
        return text
```























