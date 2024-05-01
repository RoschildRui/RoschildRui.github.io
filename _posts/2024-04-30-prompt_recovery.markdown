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
在这一篇blog中我会先简单介绍我们团队的最终提交方案**(`PV=0.6694`,`PB=0.6659`)**

随后整个方案的迭代形成过程将在我接下来的blog中逐一呈现

同时，我们也会在接下来的blog中通过复现金牌区以及部分优秀银牌区的比赛方案，反思总结，以期加深我们对于大模型的理解

感谢 [Kaggle](https://www.kaggle.com/) ——一个开放，富有活力的数据科学社区竞赛平台，为我们提供了一次深入理解LLM内部机制的机会。😉

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

### 评价指标的理解

我将通过一个具体的数学例子来展示锐化余弦相似度与传统余弦相似度之间的区别。进而更好地理解比赛评价指标的倾向性。

**示例向量**

假设我们有两对向量，一对较为相似，另一对较为不相似：

- 向量对A（较为相似）:

```math
  $\vec{u}$ = [1, 2, 3.0]
  $\vec{v}$ = [1, 2, 2.9]
```

- 向量对B（较为不相似）:

```math
  $\vec{x}$ = [1, 2, 3]
  $\vec{y}$ = [3, 2, 1] 
```

**计算余弦相似度**

余弦相似度公式为：

$\text{cosine similarity}$ = $\frac{\vec{a} \cdot \vec{b}}{\|\vec{a}\| \|\vec{b}\|}$

其中 $\vec{a} \cdot \vec{b} $是向量的点积，$\|\vec{a}\|$ 和 $\|\vec{b}\|$ 是向量的模。
 
对于向量对A和B，计算余弦相似度：

- 对A:

```math
  $ \vec{u} \cdot \vec{v} = 1*1 + 2*2 + 3*2.9 = 1 + 4 + 8.7 = 13.7 $
  $ \|\vec{u}\| = \sqrt{1^2 + 2^2 + 3^2} = \sqrt{14} $
  $ \|\vec{v}\| = \sqrt{1^2 + 2^2 + 2.9^2} \approx \sqrt{13.61} $
  $ \text{cosine similarity}_{A} = \frac{13.7}{\sqrt{14} \times \sqrt{13.61}} \approx 0.994 $
```

- 对B:

```math
  $ \vec{x} \cdot \vec{y} = 1*3 + 2*2 + 3*1 = 3 + 4 + 3 = 10 $
  $ \|\vec{x}\| = \sqrt{14}, \|\vec{y}\| = \sqrt{14} $
  $ \text{cosine similarity}_{B} = \frac{10}{14} \approx 0.535 $
```

**应用锐化处理（`p = 3`）**

锐化余弦相似度为 $`\text{cosine similarity}^p`$，这里取 `p = 3`：

- 对A:

```math
  $ \text{sharpened cosine similarity}_{A} = 0.994^3 \approx 0.982 $
```

- 对B:

```math
  $ \text{sharpened cosine similarity}_{B} = 0.535^3 \approx 0.153 $
```

**分析结果**

在上述示例中，向量对A的余弦相似度很高（接近1），通过锐化处理，它的相似度值虽未被进一步增强，但减少相对有限。对于向量对B，锐化处理显著降低了它的相似度值，向量对A与向量对B的差值显著提高了。

通过锐化余弦相似度，我们可以明显地区分高度相似、一般相似、和迥异的情况，锐化余弦相似度通过**强化已有的相似度或差异**，**使得模型预测值更加“尖锐”**。

这说明这场比赛会以较高的精度区分非常相近、一般相近、或者迥异的实体文本。进而对于我们的嵌入向量预测的准确性提出了很高的要求。**（也为我们后面集成多个模型进行预测输出埋下了伏笔）** 😎

### 数据集的构建

因为比赛方提供的数据**有限甚至等同于没有**，训练集和测试集都只有**一条**😰，所以我们需要额外查找生成一些数据以方便训练**端到端模型**以及**构建本地cv库**

#### 数据来源
以下是我们的数据来源，这里必须要感谢kaggle**开源数据集的大佬们**，以及**整理开源数据的开源大佬们**！！！🥳🥳🥳

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
我们发现数据集中存在一些无效信息，以及一些有效信息存在噪声，于是我们主要参考一位kaggle的expert预处理方案对数据集进行了**正则去噪**（抱歉，找不到那个方案的说明去哪里了，所以就让Claude写一下吧😋）

去噪内容如下：
1. 去除无用或不相关的文本:
   - 在生成的`rewritten_text`中,包含一些无用或不相关的文本,如"therefore.*I cannot"、"does not contain any"等。
   - 这些文本是由语言模型生成的,但并不是我们想要的重写后的文本。
   - 通过使用正则表达式匹配这些模式,并将匹配到的文本清空,可以去除这些无用或不相关的文本。

2. 提取相关的文本片段:
   - 在某些情况下,`rewritten_text`中包含一些固定的模式,如"Sure, here.*?:"、"Summary of .*?\n\n"等。
   - 这些模式后面通常跟着我们真正感兴趣的重写后的文本。
   - 通过使用正则表达式匹配这些子模式,并从匹配位置的末尾开始提取文本,可以获取到相关的文本片段。

3. 去除多余的前缀或后缀:
   - 有时生成的`rewritten_text`包含一些多余的前缀或后缀,如以"**\n\n"或"user"开头的文本。
   - 这些前缀或后缀是由不同语言模型的一些特性生成的,但对于重写后的文本来说是多余的。
   - 通过去掉这些多余的前缀或后缀,获得更干净和准确的重写后的文本。

4. 处理特殊情况:
   - `fix_prompt`函数中的正则表达式可以处理一些特殊情况。
   - 例如,匹配"therefore.*I cannot"模式的文本表示这个大语言模型大概率无法生成合适的重写,因此将其清空。
   - 匹配"Sure, here.*?:"模式的文本可能表示语言模型生成了一些固定的回复格式,需要提取其后的相关文本。

**参考代码如下：**
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

#### embedding数据
将上述步骤处理好的数据，通过**sentence-t5-base**模型，生成训练集和测试集的embedding

**参考代码如下：**
````python
import pandas as pd
import gc
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

df = pd.read_parquet(f"./train_clean.parquet", columns=['rewrite_prompt'])
test= pd.read_csv('./test.csv', usecols=['rewrite_prompt'])

model =  SentenceTransformer('sentence-transformers/sentence-t5-base')
model.max_seq_length = 512

encoded_data = model.encode(list(df['rewrite_prompt']), batch_size=32, device='cuda', show_progress_bar=True, convert_to_tensor=True, normalize_embeddings=True)
encoded_data = encoded_data.detach().cpu().numpy()
encoded_data = np.asarray(encoded_data.astype('float32'))

np.save('train_emb_sentence-t5.npy', encoded_data)

test_emb = model.encode(list(test['rewrite_prompt']), batch_size=32, device='cuda', show_progress_bar=True, convert_to_tensor=True, normalize_embeddings=True)
test_emb = test_emb.detach().cpu().numpy()
test_emb = np.asarray(test_emb.astype('float32'))

np.save('test_emb_sentence-t5.npy', test_emb)
```

### 训练deberta模型
**[Reference](https://www.kaggle.com/code/alejopaullier/llm-pr-seq2seq-train/notebook)**
我们主要基于这个笔记本进行了一些修改

#### 模型参数设置
```python
class config:
    AMP = True
    BATCH_SIZE_TRAIN = 4
    BATCH_SIZE_VALID = 4
    BETAS = (0.85, 0.999)
    DEBUG = 0 
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    LR = 5e-6
    EPOCHS = 10
    EPS = 1e-8
    GRADIENT_CHECKPOINTING = False
    MODEL = "/kaggle/input/deberta-v3-large-hf-weights"
    CKPT = 'deberta-v3-large'
    MAX_GRAD_NORM = 250000.0
    MAX_LEN = 384
    NUM_WORKERS = 0
    PRINT_FREQ = 500
    SEED = 42
    WANDB = False
    WEIGHT_DECAY = 0.005
```

#### 模型结构
模型结构中有两个tricks
- 1.直接使用deberta的预训练权重提取原始文本和重写文本的特征，我们发现全参数训练和使用预训练权重在测试集上的表现**没有显著的差别**，在交叉验证的时候甚至发现在某些时候会**弱于**预训练权重，于是我们决定直接使用**预训练的权重**并设计了一个**头结构**，让这个头能从底层模型deberta中提取的丰富特征中学习到有用的表示，进而通过变换和压缩，生成能够有效预测重写提示的嵌入向量。关于为什么将中间层维度设为32256，简单来说就是玄学😅，硬要说就是768*42，一般将**中间层向量维度设为嵌入向量维度的n倍**会取得不错的效果😊，这里我们直接从  `n=36`开始尝试最终发现`n=42`取得了不错的效果。（我的评价是经验，因为我们既希望模型能从deberta提取到的特征中学到**更丰富的语义信息**又希望不要**overfitting**，如果觉得太玄学，直接使用后面两个模型集成效果也足够取得不错的效果**(`PV=0.6573`,`PB=0.6569`在LB中私榜排81名，同样是银牌位)**，这个seq2seq模型就当看一个乐子了😇）
- 2.在设计的头结构中使用BatchNorm代替LayNorm（在多次交叉验证中平均涨点**0.003**），这我觉得只能算是，**四个特定**，特定任务、特定嵌入模型、特定评价指标、特定数据集的trick （我将batch_size设为2依旧如此） 🤔
```python
class CustomModel(nn.Module):
    def __init__(self, cfg, config_path=None, mode: str ="train", pretrained=False): 
        super().__init__()
        self.cfg = cfg
        self.mode = mode
        self.dropout = 0.2
        if config_path is None:
            self.config = AutoConfig.from_pretrained(cfg.MODEL, output_hidden_states=True) # output_hidden_states=True 表示在配置中启用输出隐藏状态的选项,即在前向传播过程中保存每一层的隐藏状态。
            self.config.hidden_dropout = 0.
            self.config.hidden_dropout_prob = 0. # Dropout率
            self.config.attention_dropout = 0. # 注意力权重的dropout率
            self.config.attention_probs_dropout_prob = 0.

        else:
            self.config = torch.load(config_path)

        if pretrained:
            self.model = AutoModel.from_pretrained(cfg.MODEL, config=self.config) #self.config：这是一个配置对象，包含了模型应有的配置。用来修改默认的模型配置，如调整dropout率或其他架构相关的设置。
        else:
            self.model = AutoModel(self.config)

        """
        功能：用于启用模型的梯度检查点功能，以优化内存使用，特别是在处理非常大的模型时。
        条件：只有当配置（self.cfg）中的 GRADIENT_CHECKPOINTING 属性为 True 时，才启用梯度检查点。这通常在配置文件中指定，或在实例化模型类之前设置。
        方法：gradient_checkpointing_enable() 是 transformers 库提供的方法，它允许模型在训练时仅保存必要的激活，而在需要时重新计算其他激活，从而减少内存消耗。
        """
        if self.cfg.GRADIENT_CHECKPOINTING:
            self.model.gradient_checkpointing_enable()

        self.head = nn.Sequential(
            nn.Linear(self.config.hidden_size*4, 32256),
            nn.BatchNorm1d(32256),
            nn.ReLU(),
            nn.Linear(32256, 768),
        )
        """
        self._init_weights 是一个自定义方法，用于初始化传入模块的权重。此方法通常包含针对不同类型的层（如线性层、嵌入层、层归一化等）的特定初始化策略。
        这种初始化包括设置权重的初始分布（如正态分布），并对偏置进行零初始化等。
        """
        self._init_weights(self.head)

    # 定义 _init_weights 的方法，用于自定义权重初始化。
    """
    这个 _init_weights 方法提供了一个权重初始化策略，
    针对不同类型的层采用了最适合的初始化方式。有助于模型训练的稳定性和收敛速度。
    """
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            """
            线性层（nn.Linear）：如果 module 是一个线性层，下面的方法将使用正态分布来初始化其权重，其中均值 (mean) 为 0，
            标准差 (std) 为 self.config.initializer_range。这个 initializer_range 通常是在模型的配置中指定的，用于控制初始化的分布范围。
            如果该层包含偏置 (bias)，则将偏置初始化为零。
            """
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            """
            嵌入层（nn.Embedding）：如果 module 是一个嵌入层，同样使用正态分布初始化其权重。
            如果嵌入层有 padding_idx（一般用于标记嵌入矩阵中的填充位置），则将这个位置的权重显式设置为零。这是为了确保填充位置的嵌入不会对模型产生任何影响
            """
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            """
            层归一化（nn.LayerNorm）：如果 module 是层归一化层，方法将其偏置 (bias) 初始化为零，权重 (weight) 初始化为 1。
            层归一化的权重和偏置通常用于调整归一化后数据的比例和偏移。
            """
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def feature(self, inputs):
        """
        self.model(**inputs): 调用预训练的 Transformer 模型，传入的 inputs 字典包含了诸如 input_ids, attention_mask 等键值对。
        **inputs 是 Python 的语法，表示将字典拆包为关键字参数。
        outputs: 模型的输出通常是一个包含多个组件的对象。对于许多基于 Hugging Face 的 Transformer 模型，outputs 包含了 last_hidden_state（最后一层的隐藏状态），
        hidden_states（如果配置了返回所有隐藏层的状态），以及可能的 attentions（如果配置了返回注意力权重）。
        """
        outputs = self.model(**inputs)
        #last_hidden_states = outputs[1]
        feature1 = self.pool(outputs.hidden_states[-1], inputs['attention_mask'])
        feature2 = self.pool(outputs.hidden_states[-2], inputs['attention_mask'])
        """
        torch.cat([feature1, feature2], dim=1): 将 feature1 和 feature2 沿着第二维度（即特征维度）拼接起来。
        这种方式融合来自最后两个隐藏层的信息，使得生成的特征向量不仅包含了最终层的上下文信息，也融入了之前层的语义特征。
        返回的结果是一个扩展的特征向量，现在的特征大小是原来每层特征大小的两倍，因为它包含了两层的输出。
        """
        return torch.cat([feature1, feature2], dim=1)

    def forward(self, original_texts, rewritten_texts, rewrite_prompts_embedding):

        original_texts_feature = self.feature(original_texts) # shape (batch_size, 768)
        rewritten_texts_feature = self.feature(rewritten_texts) # shape (batch_size, 768)
        feature = torch.cat([original_texts_feature, rewritten_texts_feature], dim=1) # shape (batch_size, 768 * 2)
        output = self.head(feature)

        if self.mode == "train":
            prompt_embedding = torch.tensor(rewrite_prompts_embedding, device=self.cfg.DEVICE) # shape (batch_size, 768)
        else:
            prompt_embedding = None

        return output, prompt_embedding
```





















