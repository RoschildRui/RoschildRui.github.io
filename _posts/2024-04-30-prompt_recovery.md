---
title: "[LLM Prompt Recovery](PV:0.67)银牌方案总结(1)"
math: true
---
> “这是我的第一个blog，希望在这一路上能认识更多志同道合的朋友”

> "This is my first blog post. Setting out on this path, I am looking forward to meeting more friends who also have greate passion on data science. Together, we will dive into the fascinating world of data, uncover valuable and deeper insights on this Earth, and strive to make a meaningful impact to make people life great 'again' !!!"

### 写在前面
在这一篇blog中我会先简单介绍我们团队的最终提交方案**(`PV=0.67`,`PB=0.66`)**

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


### 方案预览
![image](https://github.com/RoschildRui/RoschildRui.github.io/assets/146306438/d43d75a0-7093-4d06-904d-cbd9b3db61a2)

- 1.构建基于deberta-v3-large的seq2seq模型
- 2.构建合适的adapter层微调[phi2](https://www.kaggle.com/models/Microsoft/phi/Transformers/2/1)模型
- 3.few-shot [mistral-7b-v2](https://www.kaggle.com/datasets/ahmadsaladin/mistral-7b-it-v02)模型

将三个模型还原的提示词集成在一起作为最终的预测结果

### 评价指标的理解

我将通过一个具体的数学例子来展示锐化余弦相似度与传统余弦相似度之间的区别。进而更好地理解比赛评价指标的倾向性。

**示例向量**

假设我们有两对向量，一对较为相似，另一对较为不相似：

- 向量对A（相似）:

```math
  $\vec{u}$ = [1, 2, 3.0]
  $\vec{v}$ = [1, 2, 2.9]
```

- 向量对B（不相似）:

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

通过锐化余弦相似度，我们可以明显地区分高度相似、一般相似、和迥异的情况，锐化余弦相似度通过**强化已有的相似度的差异**，**使得模型预测值更加“尖锐”**。

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
我们发现数据集中存在一些无效信息，以及一些有效信息存在噪声，于是我们主要参考一位kaggle expert预处理方案对数据集进行了**正则去噪**（抱歉，找不到那个方案的说明去哪里了，所以就让Claude写一下吧😋）

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

同时将上述数据中的unique prompt提示词整理为一个prompt文件，利用Meta开源的Faiss库将其转为prompt.index，对deberta模型预测结果进行相似度匹配进而输出**（关于为什么用这种方法这里只讲一点剩下的后面会详细讲解-----这种方法涨点显著但是要求私有数据集构建完善，我们通过使用提示词工程调用gpt-4生成了150条高质量平均提示词（`PB=0.58`以上，同时在开源的数据集中找到了1400000余条富有特征的提示词,没错就是140w条你没看错😀）**

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

```python
import sys
from sentence_transformers import SentenceTransformer, models
import pandas as pd
import gc
import numpy as np
import faiss
import time
from tqdm import tqdm

df = pd.read_csv(f"prompts_df.csv",)
contexts = list(df['rewrite_prompt'])

model =  SentenceTransformer('sentence-transformers/sentence-t5-base')
model.max_seq_length = 512

encoded_data = model.encode(contexts, batch_size=32, device='cuda', show_progress_bar=True, convert_to_tensor=True, normalize_embeddings=True)
encoded_data = encoded_data.detach().cpu().numpy()
encoded_data = np.asarray(encoded_data.astype('float32'))
df['rewrite_prompt'].to_csv('prompts_df.csv', index=False)

index = faiss.IndexFlatIP(768)
index.add(encoded_data)
faiss.write_index(index, 'prompts_embedding.index')
```

### 训练deberta模型
**[Reference](https://www.kaggle.com/code/alejopaullier/llm-pr-seq2seq-train/notebook)**
我们主要基于这个笔记本进行了一些提升修改
#### 模型参数设置
```python
class config:
    AMP = True
    BATCH_SIZE_TRAIN = 4
    BATCH_SIZE_VALID = 4
    BETAS = (0.9, 0.999)
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

参考代码如下：
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

#### 模型训练
上述deberta模型的训练在Autodl的4090上**一个epoch**大约在**6个小时左右**

![image](https://github.com/RoschildRui/RoschildRui.github.io/assets/146306438/c10000e5-7c73-4b27-8d28-fdabf457d8c7)

我们发现当`batch_size=2`、不打开GRADIENT_CHECKPOINTING是比 `batch_size=16`并打开GRADIENT_CHECKPOINTING快，并且**测试集评估效果没有显著的影响**（有些时候提升了），于是我们选择不打开checkpoint

#### 模型推理
将模型预测结果与之前embedding的index进行匹配，选择**相似度最高**的作为预测结果进行填充

参考代码如下：
```python
def inference_fn(model_weight, config, test_df, tokenizer, device, model_config):
    # ======== DATASETS ==========
    test_dataset = CustomDataset(config, test_df, tokenizer)
    
    # ======== DATALOADERS ==========
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE_TEST,
        shuffle=False,
        num_workers=0,
        pin_memory=True, drop_last=False
    )
    
    # ======== MODEL ==========
    model = CustomModel(config, config_path=model_config, pretrained=False)
    state = torch.load(model_weight)
    model.load_state_dict(state)
    model.to(device)
    model.eval() # set model in evaluation mode
    output_dict = {}
    preds, ids = [], []
    with tqdm(test_loader, unit="test_batch", desc='Test') as tqdm_test_loader:
        for step, batch in enumerate(tqdm_test_loader):
            ids_batch = batch.pop("id")
            original_texts = to_device(collate(batch.pop("original_text")))
            rewritten_texts = to_device(collate(batch.pop("rewritten_text")))
            rewrite_prompts = []
            batch_size = len(ids_batch)
            targets = torch.ones(batch_size, device=device) # -1 for dissimilar, 1 for similar
            with torch.no_grad():
                y_preds, _ = model(original_texts, rewritten_texts, rewrite_prompts)            
            preds.append(y_preds.to('cpu').numpy()) # save predictions
            ids += ids_batch          
    output_dict["predictions"] = np.concatenate(preds) 
    output_dict["ids"] = ids
    return output_dict

preds = []

for model_weight, model_config in zip(model_weights, model_configs):
    predictions = inference_fn(model_weight, config, test_df, tokenizer, device, model_config)
    predictions = predictions["predictions"]
    predictions = torch.nn.functional.normalize(torch.from_numpy(predictions), p=2, dim=1).numpy()
    preds.append(predictions)
    
preds = np.mean(preds, axis=0)

import faiss
from faiss import write_index, read_index, read_VectorTransform

prompts_embedding_index = read_index("./prompts_embedding.index")
search_score, search_index = prompts_embedding_index.search(preds, 1)
prompts_df = pd.read_csv("./prompts_df.csv")
prompts_df.head()

pred_prompts = []

for i, (scr, idx) in tqdm(enumerate(zip(search_score, search_index)), total=len(search_score)):
    scr_idx = idx
    p = prompts_df.loc[scr_idx, "rewrite_prompt"].tolist()
    pred_prompts.append(''.join(p))

values = pred_prompts

submission = pd.DataFrame()
submission["id"] = test_df["id"]
submission["rewrite_prompt"] = values
submission.to_csv("submission_1.csv", index=False)
```

好了终于把第一个模型写完了
![image](https://github.com/RoschildRui/RoschildRui.github.io/assets/146306438/509cd940-f1b4-4e54-a19b-5a010aa0a38e)

### 微调[phi](https://www.kaggle.com/models/Microsoft/phi/Transformers/2/1)
思路来源于这位大佬开源的[Notebook1](https://www.kaggle.com/code/mozhiwenmzw/0-61-llmpr-phi2-sft-model-generate-infer/notebook)和[Notebook2](https://www.kaggle.com/code/mozhiwenmzw/0-61-llmpr-phi2-sft-model-training/notebook)

同时感谢这位大佬开源的[Mean Prompt](https://www.kaggle.com/code/seifachour12/lb-score-0-63)

#### 训练adapter
在看完大佬的笔记本后，我们先尝试通过我们自己的私有数据集训练phi的adapter层进而使得它对于这个任务更加适用

参考代码如下：
```python
import pandas as pd
from sklearn.model_selection import train_test_split

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import TrainingArguments

from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig

exp_name = 'phi2'
data_path = '/kaggle/input/pr-data/train_clean.csv'
model_path = '/kaggle/input/phi/transformers/2/1'
output_path = f'outputs'
model_save_path =  f'{exp_name}_adapter'

epochs=5
batch_size=1 # 2 
max_seq_length=512 # 1024 
lr = 1e-4

df = pd.read_csv(data_path)
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

train_ds = Dataset.from_pandas(train_df)
val_ds = Dataset.from_pandas(val_df)

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype='float16',
        bnb_4bit_use_double_quant=False,
    )

model = AutoModelForCausalLM.from_pretrained(model_path,
                                             quantization_config=bnb_config,
                                             trust_remote_code=True,
                                             use_auth_token=True)

model.config.gradient_checkpointing = False

def token_len(text):
    tokenized = tokenizer(text, return_length=True)
    length = tokenized['length'][0]
    return length

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['rewritten_text'])):
        ori_text = example['original_text'][i]
        rew_text = example['rewritten_text'][i]
        rew_prompt = example['rewrite_prompt'][i]
        text = f"Instruct: Original Text:{ori_text}\nRewritten Text:{rew_text}\nWrite a prompt that was likely given to the LLM to rewrite original text into rewritten text.Output: {rew_prompt}"
        if token_len(text) > max_seq_length:
            continue
        output_texts.append(text)
    return output_texts

response_template = "Output:"
collator = DataCollatorForCompletionOnlyLM(response_template=response_template, 
                                           tokenizer=tokenizer)

peft_config = LoraConfig(
    r=12,
    lora_alpha=32,
    lora_dropout=0.03,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules= ["q_proj", "k_proj", "v_proj", "dense"],
)

args = TrainingArguments(
    output_dir = output_path,
    fp16=True,
    learning_rate=lr,
    optim="adafactor",
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size*2,
    gradient_accumulation_steps=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    logging_steps=50,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    weight_decay=0.008,
    report_to='none',
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    )

trainer = SFTTrainer(
    model=model,
    args = args,
    max_seq_length=max_seq_length,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    formatting_func=formatting_prompts_func,
    data_collator=collator,
    peft_config=peft_config,
)

trainer.train()

trainer.save_model(model_save_path)
tokenizer.save_pretrained(model_save_path)
```
#### 利用adapter微调phi
但是我们发现不管在phi模型的顶层还是中间层训练adapter似乎都无法达到大佬[开源版本](https://www.kaggle.com/models/mozhiwenmzw/phi2-public-data-sft-adapter/PyTorch/public-data-sft/1)的效果（单模最高能到`PB=0.63`但是集成就会使得PB相对使用开源的adapter下降0.1左右）😅

所以最后我们直接使用开源的adapter进行微调phi

我们对开源代码上进行了一些调整以针对我们最后的集成方案进行优化
- 根据对生成文本的观察，添加符号'.',';',':','<endoftext>' （抱歉，渲染出来的格式好像有点问题😳） 作为生成文本的停止标记，进而严格控制phi2生成文本的时间
- 去除生成文本最后的符号，我们发现在集成预测结果的时候要严格控制句号的数目，去掉句号能在PB提高0.01分左右 🤠

参考代码如下：
```python
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

input_token_len = 1024
output_token_len = 100
test_df = pd.read_csv('/kaggle/input/llm-prompt-recovery/test.csv')
base_model_name = "/kaggle/input/phi/transformers/2/1"
adapter_model_name = "/kaggle/input/phi2-public-data-sft-adapter/pytorch/public-data-sft/1/phi2_public_data_sft/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(base_model_name,trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    
model = AutoModelForCausalLM.from_pretrained(base_model_name,trust_remote_code=True)
model = PeftModel.from_pretrained(model, adapter_model_name)
model.to(device)
model.eval()

def text_generate(ori_text, rew_text,model, tokenizer, stop_tokens=['.',';',':','<|endoftext|>'], input_max_len=512, output_len=20, device='cuda'):
    prompt = f"Instruct: Original Text:{ori_text}\nRewritten Text:{rew_text}\nWrite a prompt that was likely given to the LLM to rewrite original text to rewritten text.\nOutput:"
    inputs = tokenizer(prompt, max_length=input_max_len, truncation=True, return_tensors="pt", return_attention_mask=False)
    output_start_index = len(inputs.input_ids[0])
    inputs = {k:v.to(device) for k,v in inputs.items()}
    outputs = model.generate(**inputs,
                             do_sample=False,
                             max_new_tokens=output_len,
                             pad_token_id=tokenizer.pad_token_id,
                             eos_token_id=tokenizer.convert_tokens_to_ids(stop_tokens),
                            )
    text = tokenizer.batch_decode(outputs,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
    start_index = text.find('Output:')
    generated_text = text[start_index+len('Output:'):].strip()[:-1]
    return generated_text

rewrite_prompts = []
for i, row in tqdm(test_df.iterrows(), total=len(test_df)):
    prompt = mean_prompt = 'Please improve this text.'
    # 因为集成的原因所以这里不使用'Please improve this text using the writing style with maintaining the original meaning but altering the tone.'单模使用这个
    try:
        prompt = text_generate(row['original_text'],
                               row['rewritten_text'],
                               model,
                               tokenizer,
                               ['.',';',':','<|endoftext|>'],
                               input_token_len,
                               output_token_len,
                               device,
                              )
    except:
        pass
        
    rewrite_prompts.append(prompt)

test_df['rewrite_prompt'] = rewrite_prompts
sub_df = test_df[['id', 'rewrite_prompt']]
sub_df.to_csv('submission_2.csv', index=False)
```

有点稳了想水一下🫠

### few-shot mistral-7b模型
这个应该是比赛中最火爆的方案，无论开源还是闭源

同样，这里感谢一下开源的[方案](https://www.kaggle.com/code/richolson/mistral-7b-prompt-recovery-version-2)分享🥳

#### few-shot限制条件
使用few-shot方法来训练大型模型完成提示词还原任务有一些严格的限制条件：
- 1. 需要模型具有较高的基准能力/这场比赛中few-shot llama2-13b与few-show mistral-7b 在PB的差距约为`0.02`，在PV的差距在`0.013`以内
![image](https://github.com/RoschildRui/RoschildRui.github.io/assets/146306438/ec1cf9ea-ab7d-4bcb-b8a4-79da3c8fdc4d)
上面的图是mistral官方提供的benchmarks对比图，我们认为这个任务主要需要大模型的核心能力为**Reasoning**、**Knowledge**、**Comprehension**
- 2. 需要选取具有代表性和多样性的示例作为few-shot提示数据。这些示例需要能够涵盖任务可能的不同变体和复杂性，且要清晰传达任务的需求。庆幸的是，我们在众多discussion中学习提炼创造了多样可行的示例并与开源的示例加以结合实现了PB提升`0.01`，PV提升了`0.0072`（下面代码中将不会有我们自己构建的示例，因为我们对其中的一些点存在疑惑，我们将会在解决疑惑后进行开源分享）😛
- 3. 需要设计合适的提示词，使得大模型更加关注对于某一些特征的提取

#### few-shot
我们基于开源的方案做了3点改进：
- 1.将`response_prefix = "Improve this text by"`改为`response_prefix = "Improve this text using the writing style"`使得模型更加关注writing style的特征提取
- 2.指定大模型输出的`stop_tokens = ['.',':']`
- 3.如果生成文本的长度低于15，使用PB高分Mean prompt作为替换，在所有未被替换的提示词末尾中添加`', maintaining the original meaning but altering the tone.'`

参考代码如下：
```python
import torch
import random
import numpy as np
import pandas as pd
import gc
import time

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

if (not torch.cuda.is_available()): print("Sorry - GPU required!")
    
import logging
logging.getLogger('transformers').setLevel(logging.ERROR)
#this can help speed up inference
max_new_tokens = 30

#output test is trimmed according to this
max_sentences_in_response = 1
model_name = '/kaggle/input/mistral-7b-it-v02'
tokenizer = AutoTokenizer.from_pretrained(model_name) 

# Load base model(Mistral 7B)
bnb_config = BitsAndBytesConfig(  
    load_in_4bit= True,
    bnb_4bit_quant_type= "nf4",
    bnb_4bit_compute_dtype= torch.bfloat16,
    bnb_4bit_use_double_quant= False,
)

model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
)
#original text prefix
orig_prefix = "Original Text:"

#mistral "response"
llm_response_for_rewrite = "Provide the new text and I will tell you what new element was added or change in tone was made to improve it - with no references to the original.  I will avoid mentioning names of characters.  It is crucial no person, place or thing from the original text be mentioned.  For example - I will not say things like 'change the puppet show into a book report' - I would just say 'Please improve this text using the writing style of a book report'.  If the original text mentions a specific idea, person, place, or thing - I will not mention it in my answer.  For example if there is a 'dog' or 'office' in the original text - the word 'dog' or 'office' must not be in my response.  My answer will be a single sentence."

#modified text prefix
rewrite_prefix = "Re-written Text:"

#provided as start of Mistral response (anything after this is used as the prompt)
#providing this as the start of the response helps keep things relevant
response_start = "The request was: "

#added after response_start to prime mistral
#"Improve this" or "Improve this text" resulted in non-answers.  
#"Improve this text by" seems to product good results
response_prefix = "Please improve this text using the writing style"

#well-scoring baseline text
#thanks to: https://www.kaggle.com/code/rdxsun/lb-0-61
base_line = 'Please improve this text using the writing style with maintaining the original meaning but altering the tone.' 

#these will all be given to Mistral before each and every prompt
#original_text
#rewritten_text
#prompt

examples_sequences = [
    (
        "Hey there! Just a heads up: our friendly dog may bark a bit, but don't worry, he's all bark and no bite!",
        "Warning: Protective dog on premises. May exhibit aggressive behavior. Ensure personal safety by maintaining distance and avoiding direct contact.",
        "Please improve this text using the writing style of a warning."
    ),

    (
        "A lunar eclipse happens when Earth casts its shadow on the moon during a full moon. The moon appears reddish because Earth's atmosphere scatters sunlight, some of which refracts onto the moon's surface. Total eclipses see the moon entirely in Earth's shadow; partial ones occur when only part of the moon is shadowed.",
        "Yo check it, when the Earth steps in, takes its place, casting shadows on the moon's face. It's a full moon night, the scene's set right, for a lunar eclipse, a celestial sight. The moon turns red, ain't no dread, it's just Earth's atmosphere playing with sunlight's thread, scattering colors, bending light, onto the moon's surface, making the night bright. Total eclipse, the moon's fully in the dark, covered by Earth's shadow, making its mark. But when it's partial, not all is shadowed, just a piece of the moon, slightly furrowed. So that's the rap, the lunar eclipse track, a dance of shadows, with no slack. Earth, moon, and sun, in a cosmic play, creating the spectacle we see today.",
        "Please improve this text using the writing style of a rap."
    ),
    
    (
        "Drinking enough water each day is crucial for many functions in the body, such as regulating temperature, keeping joints lubricated, preventing infections, delivering nutrients to cells, and keeping organs functioning properly. Being well-hydrated also improves sleep quality, cognition, and mood.",
        "Arrr, crew! Sail the health seas with water, the ultimate treasure! It steadies yer body's ship, fights off plagues, and keeps yer mind sharp. Hydrate or walk the plank into the abyss of ill health. Let's hoist our bottles high and drink to the horizon of well-being!",
        "Please improve this text using the writing style of a sea pirate."
    ),
    
    (
        "In a bustling cityscape, under the glow of neon signs, Anna found herself at the crossroads of endless possibilities. The night was young, and the streets hummed with the energy of life. Drawn by the allure of the unknown, she wandered through the maze of alleys and boulevards, each turn revealing a new facet of the city's soul. It was here, amidst the symphony of urban existence, that Anna discovered the magic hidden in plain sight, the stories and dreams that thrived in the shadows of skyscrapers.",
        "On an ordinary evening, amidst the cacophony of a neon-lit city, Anna stumbled upon an anomaly - a door that defied the laws of time and space. With the curiosity of a cat, she stepped through, leaving the familiar behind. Suddenly, she was adrift in the stream of time, witnessing the city's transformation from past to future, its buildings rising and falling like the breaths of a sleeping giant.",
        "Please improve this text using the writing style with time travel topic."
    ),
    
    (
        "Late one night in the research lab, Dr. Evelyn Archer was on the brink of a breakthrough in artificial intelligence. Her fingers danced across the keyboard, inputting the final commands into the system. The lab was silent except for the hum of machinery and the occasional beep of computers. It was in this quiet orchestra of technology that Evelyn felt most at home, on the cusp of unveiling a creation that could change the world.",
        "In the deep silence of the lab, under the watchful gaze of the moon, Dr. Evelyn Archer found herself not alone. Beside her, the iconic red eye of HAL 9000 flickered to life, a silent partner in her nocturnal endeavor. 'Good evening, Dr. Archer,' HAL's voice filled the room, devoid of warmth yet comforting in its familiarity. Together, they were about to initiate a test that would intertwine the destiny of human and artificial intelligence forever. As Evelyn entered the final command, HAL processed the data with unparalleled precision, a testament to the dawn of a new era.",
        "Please improve this text using the writing style with an intelligent computer."
    ),
    
    (
        "The park was empty, save for a solitary figure sitting on a bench, lost in thought. The quiet of the evening was punctuated only by the occasional rustle of leaves, offering a moment of peace in the chaos of city life.",
        "Beneath the cloak of twilight, the park transformed into a realm of solitude and reflection. There, seated upon an ancient bench, was a lone soul, a guardian of secrets, enveloped in the serenity of nature's whispers. The dance of the leaves in the gentle breeze sang a lullaby to the tumult of the urban heart.",
        "Please improve this text using the writing style to be more poetic."
    ),
    
    (
        "The annual town fair was bustling with activity, from the merry-go-round spinning with laughter to the game booths challenging eager participants. Amidst the excitement, a figure in a cloak moved silently, almost invisibly, among the crowd, observing everything with keen interest but participating in none.",
        "Beneath the riot of color and sound that marked the town's annual fair, a solitary figure roamed, known to the few as Eldrin the Enigmatic. Clad in a cloak that shimmered with the whispers of the arcane, Eldrin moved with the grace of a shadow, his gaze piercing the veneer of festivity to the magic beneath. As a master of the mystic arts, he sought not the laughter of the crowds but the silent stories woven into the fabric of the fair. With a flick of his wrist, he could coax wonder from the mundane, transforming the ordinary into spectacles of shimmering illusion, his true participation hidden within the folds of mystery.",
        "Please improve this text using the writing style by adding a magician."
    ),
    
    (
        "The startup team sat in the dimly lit room, surrounded by whiteboards filled with ideas, charts, and plans. They were on the brink of launching a new app designed to make home maintenance effortless for homeowners. The app would connect users with local service providers, using a sophisticated algorithm to match needs with skills and availability. As they debated the features and marketing strategies, the room felt charged with the energy of creation and the anticipation of what was to come.",
        "In the quiet before dawn, a small group of innovators gathered, their mission: to simplify home maintenance through technology. But their true journey began with the unexpected addition of Max, a talking car with a knack for solving problems. 'Let me guide you through this maze of decisions,' Max offered, his dashboard flickering to life.",
        "Please improve this text using the writing style by adding a talking car."
    ),
]

def remove_numbered_list(text):
    final_text_paragraphs = [] 
    for line in text.split('\n'):
        # Split each line at the first occurrence of '. '
        parts = line.split('. ', 1)
        # If the line looks like a numbered list item, remove the numbering
        if len(parts) > 1 and parts[0].isdigit():
            final_text_paragraphs.append(parts[1])
        else:
            # If it doesn't look like a numbered list item, include the line as is
            final_text_paragraphs.append(line)

    return '  '.join(final_text_paragraphs)

#trims LLM output to just the response
def trim_to_response(text):
    terminate_string = "[/INST]"
    text = text.replace('</s>', '')
    #just in case it puts things in quotes
    text = text.replace('"', '')
    text = text.replace("'", '')

    last_pos = text.rfind(terminate_string)
    return text[last_pos + len(terminate_string):] if last_pos != -1 else text

#looks for response_start / returns only text that occurs after
def extract_text_after_response_start(full_text):
    parts = full_text.rsplit(response_start, 1)  # Split from the right, ensuring only the last occurrence is considered
    if len(parts) > 1:
        return parts[1].strip()  # Return text after the last occurrence of response_start
    else:
        return full_text  # Return the original text if response_start is not found

    
#trims text to requested number of sentences (or first LF or double-space sequence)
def trim_to_first_x_sentences_or_lf(text, x):
    if x <= 0:
        return ""

    # Any double-spaces dealt with as linefeed
    text = text.replace("  ", "\n")

    # Split text at the first linefeed
    text_chunks = text.split('\n', 1)
    first_chunk = text_chunks[0]

    # Split the first chunk into sentences, considering the space after each period
    sentences = [sentence.strip() for sentence in first_chunk.split('.') if sentence]

    # If there's a linefeed, return the text up to the first linefeed
    if len(text_chunks) > 1:
        # Check if the first chunk has fewer sentences than x, and if so, just return it
        if len(sentences) < x:
            trimmed_text = first_chunk
        else:
            # Otherwise, trim to x sentences within the first chunk
            trimmed_text = '. '.join(sentences[:x]).strip()
    else:
        # If there's no linefeed, determine if the number of sentences is less than or equal to x
        if len(sentences) <= x:
            trimmed_text = '. '.join(sentences).strip()  # Ensure space is preserved after periods
        else:
            # Otherwise, return the first x sentences, again ensuring space after periods
            trimmed_text = '. '.join(sentences[:x]).strip()

    # Add back the final period if it was removed and the text needs to end with a sentence.
    if len(sentences) > 0 and not trimmed_text.endswith('.'):
        trimmed_text += '.'

    return trimmed_text

def get_prompt(orig_text, transformed_text):
    stop_tokens = ['.',':']
    messages = []

    # Append example sequences
    for example_text, example_rewrite, example_prompt in examples_sequences:
        messages.append({"role": "user", "content": f"{orig_prefix} {example_text}"})
        messages.append({"role": "assistant", "content": llm_response_for_rewrite})
        messages.append({"role": "user", "content": f"{rewrite_prefix} {example_rewrite}"})
        messages.append({"role": "assistant", "content": f"{response_start} {example_prompt}"})

    #actual prompt
    messages.append({"role": "user", "content": f"{orig_prefix} {orig_text}"})
    messages.append({"role": "assistant", "content": llm_response_for_rewrite})
    messages.append({"role": "user", "content": f"{rewrite_prefix} {transformed_text}"})
    messages.append({"role": "assistant", "content": f"{response_start}"})
        
    #give it to Mistral
    decode_ids = tokenizer.encode(response_prefix, add_special_tokens=False)
    model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
    
    output_start_index = len(model_inputs[0])
    force_decoder_ids = []
    for i, did in enumerate(decode_ids):
        force_decoder_ids.append([i+output_start_index, did])
    
    model_inputs = model_inputs.to("cuda") 
    generated_ids = model.generate(model_inputs, max_new_tokens=max_new_tokens, 
                                   pad_token_id=tokenizer.eos_token_id,
                                   eos_token_id=tokenizer.convert_tokens_to_ids(stop_tokens),
                                   forced_decoder_ids = force_decoder_ids,
                                  )

    #decode and trim to actual response
    decoded = tokenizer.batch_decode(generated_ids)
    just_response = trim_to_response(decoded[0])        
    final_text = extract_text_after_response_start(just_response)
        
    #mistral has been replying with numbered lists - clean them up....
    final_text = remove_numbered_list(final_text)
        
    #mistral v02 tends to respond with the input after providing the answer - this tries to trim that down
    final_text = trim_to_first_x_sentences_or_lf(final_text, max_sentences_in_response)
    
    #default to baseline if empty or unusually short
    if len(final_text) < 15:
        final_text = base_line
        return final_text
    final_text = final_text[:-1] + ', maintaining the original meaning but altering the tone.'
    return final_text

test_df = pd.read_csv("/kaggle/input/llm-prompt-recovery/test.csv")

for index, row in test_df.iterrows():
    result = get_prompt(row['original_text'], row['rewritten_text'])
    print(result)
    test_df.at[index, 'rewrite_prompt'] = result
    
test_df = test_df[['id', 'rewrite_prompt']]
test_df.to_csv('submission_3.csv', index=False)
```

### 集成三个模型
我们将三个模型的预测结果进行拼接最终得到最后的结果🤗
```python
import pandas as pd
sub_1 = pd.read_csv('submission_1.csv').sort_values(['id']).reset_index(drop=True).fillna('')
sub_2 = pd.read_csv('submission_2.csv').sort_values(['id']).reset_index(drop=True).fillna('')
sub_3 = pd.read_csv('submisiion_3.csv').sort_values(['id']).reset_index(drop=True).fillna('')
sub = pd.DataFrame()
sub['rewrite_prompt'] = sub_1['rewrite_prompt'].map(str)+' '+sub_2['rewrite_prompt'].map(str)+' '+sub_3['rewrite_prompt'].map(str)
print(sub['rewrite_prompt'].iloc[0])
sub.to_csv('submission.csv', index=False)
```

### 总结
**完成方案总结**

在反思为什么集成三个模型的预测结果会效果好的时候，我从私有数据集中的数据提炼30条数据，分别使用3个模型（这三个模型的`PB`分数十分接近，`deberta`的`PB=0.64`，`phi`的`PB=0.65`,`mistral`的`PB=0.65`)进行预测，他们的预测结果都存在一些很明显的差异，也就是说他们生成的句子的某些特征是正交的，于是我想可能是由于这些特征在经过`sentence-t5`的encode后拉近了与目标提示词的距离，进而提高了`PB`分数。

在比赛结束后我从第一的方案分享中看到了这样一幅图
![image](https://github.com/RoschildRui/RoschildRui.github.io/assets/146306438/e4d2d756-cf67-4fd8-8b97-faafaba2d251)
通过这幅图我认为我对于集成方案的理解一定程度上是可解释的😎

**比赛意义总结**
- 1.提示词还原本质上我认为是通过模型的输出和输入去寻找最具有效率的提示方法，我认为这就是在引导我们去逐步发掘如何在未来的时代中更好的利用大模型来**高效赋能各行各业**，而不是简单粗暴的堆叠算力。
- 2.我感觉我们的大脑运行与大模型其实有一定的相似，在寻找最具有效率的提示方法的同时，也是在完成一次**教育领域的探索**---如何高效的使得受教育者在教育者高效提示词的引导下从课本上的知识（原有文本）产出教育者的期望输出（重写文本）。

### 写在最后
完整读完的朋友应该发现了这次比赛我们的模型几乎完全是在**开源**模型代码的基础上完成的，唯一有一些不同的就在于我们最后选择了将不同的模型预测结果进行了直接拼接，而不是一般的集成输出。

我想，这可能就是kaggle这个平台**最具有魅力**之处。

**kaggle**提供了一个平台使得全球热爱数据科学，机器学习的朋友能够在此相遇，这里**不论新老**，**不论强弱**，**自由交流**，让所有kaggler能够将所学知识灵活应用于前沿应用场景，也能在不断的交流中加深对所学知识的理解，学到更多新的东西。

再次感谢**kaggle**以及**kaggle上热情交流的朋友们**！！！😉

最后，感谢我的父母、学校为我提供了优越的学习研究条件，感谢吴哥zzy带我走上kaggle之路












