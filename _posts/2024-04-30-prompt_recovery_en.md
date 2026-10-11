---
title: "[LLM Prompt Recovery] (PV: 0.67) Silver Medal Solution Summary (Part 1)"
---
> "This is my first blog post. Setting out on this path, I am looking forward to meeting more friends who also have greate passion on data science. Together, we will dive into the fascinating world of data, uncover valuable and deeper insights on this Earth, and strive to make a meaningful impact to make people life great 'again' !!!"

### Preface
In this blog post I'll first briefly introduce our team's final submission **(`PV=0.67`,`PB=0.66`)**

After that, the full iterative process of how the solution took shape will be presented step by step in my upcoming posts

Meanwhile, in the upcoming posts we'll also reproduce the solutions from the gold medal zone and some of the excellent silver medal ones, reflect on them and summarize what we learn, hoping to deepen our understanding of large models

Thanks to [Kaggle](https://www.kaggle.com/), an open and vibrant data science community and competition platform, for giving us a chance to gain a deep understanding of the inner workings of LLMs.😉

### [Competition](https://www.kaggle.com/competitions/llm-prompt-recovery)
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


### Solution Overview
![image](https://github.com/RoschildRui/RoschildRui.github.io/assets/146306438/d43d75a0-7093-4d06-904d-cbd9b3db61a2)

- 1.Build a seq2seq model based on deberta-v3-large
- 2.Fine-tune the [phi2](https://www.kaggle.com/models/Microsoft/phi/Transformers/2/1) model with a suitable adapter layer
- 3.A few-shot [mistral-7b-v2](https://www.kaggle.com/datasets/ahmadsaladin/mistral-7b-it-v02) model

The prompts recovered by the three models are ensembled together as the final prediction

### Understanding the Evaluation Metric

I'll use a concrete mathematical example to show the difference between sharpened cosine similarity and traditional cosine similarity, so we can better understand what the competition metric favors.

**Example Vectors**

Suppose we have two pairs of vectors, one fairly similar and the other fairly dissimilar:

- Vector pair A (similar):

```math
  $\vec{u}$ = [1, 2, 3.0]
  $\vec{v}$ = [1, 2, 2.9]
```

- Vector pair B (dissimilar):

```math
  $\vec{x}$ = [1, 2, 3]
  $\vec{y}$ = [3, 2, 1] 
```

**Computing Cosine Similarity**

The cosine similarity formula is:

$\text{cosine similarity}$ = $\frac{\vec{a} \cdot \vec{b}}{\|\vec{a}\| \|\vec{b}\|}$

where $\vec{a} \cdot \vec{b} $ is the dot product of the vectors, and $\|\vec{a}\|$ and $\|\vec{b}\|$ are their magnitudes.

For vector pairs A and B, compute the cosine similarity:

- For A:

```math
  $ \vec{u} \cdot \vec{v} = 1*1 + 2*2 + 3*2.9 = 1 + 4 + 8.7 = 13.7 $
  $ \|\vec{u}\| = \sqrt{1^2 + 2^2 + 3^2} = \sqrt{14} $
  $ \|\vec{v}\| = \sqrt{1^2 + 2^2 + 2.9^2} \approx \sqrt{13.61} $
  $ \text{cosine similarity}_{A} = \frac{13.7}{\sqrt{14} \times \sqrt{13.61}} \approx 0.994 $
```

- For B:

```math
  $ \vec{x} \cdot \vec{y} = 1*3 + 2*2 + 3*1 = 3 + 4 + 3 = 10 $
  $ \|\vec{x}\| = \sqrt{14}, \|\vec{y}\| = \sqrt{14} $
  $ \text{cosine similarity}_{B} = \frac{10}{14} \approx 0.535 $
```

**Applying Sharpening (`p = 3`)**

The sharpened cosine similarity is $`\text{cosine similarity}^p`$, where we take `p = 3`:

- For A:

```math
  $ \text{sharpened cosine similarity}_{A} = 0.994^3 \approx 0.982 $
```

- For B:

```math
  $ \text{sharpened cosine similarity}_{B} = 0.535^3 \approx 0.153 $
```

**Analyzing the Results**

In the example above, vector pair A has a very high cosine similarity (close to 1). After sharpening, its similarity is not boosted any further, but the reduction is relatively limited. For vector pair B, sharpening significantly lowers its similarity, so the gap between pair A and pair B becomes much larger.

With sharpened cosine similarity, we can clearly distinguish between highly similar, moderately similar, and very different cases. Sharpened cosine similarity works by **amplifying the existing differences in similarity**, **making the model's predicted values "sharper"**.

This means the competition distinguishes, with fairly high precision, between texts that are very close, moderately close, or completely different, which in turn places very high demands on the accuracy of our predicted embedding vectors. **(This also foreshadows why we later ensemble multiple models for the final output)** 😎

### Building the Dataset

Since the data provided by the organizers was **limited, practically nonexistent**, with only **one** sample each in the training set and the test set😰, we had to find and generate extra data to train **end-to-end models** and to **build a local CV set**

#### Data Sources
Below are our data sources. A huge thank you to the **experts who open-sourced datasets** on kaggle, and to the **open-source heroes who collected and organized open data**!!!🥳🥳🥳

**reference**
- [https://www.kaggle.com/code/tomooinubushi/all-in-one-dataset-with-embedding/notebook](https://huggingface.co/datasets/Skylion007/openwebtext)
- [https://huggingface.co/datasets/Skylion007/openwebtext](https://huggingface.co/datasets/Skylion007/openwebtext)
- [https://huggingface.co/datasets/euclaise/writingprompts/viewer/default/train?p=1](https://huggingface.co/datasets/euclaise/writingprompts/viewer/default/train?p=1)
- [https://www.kaggle.com/datasets/nbroad/gemma-rewrite-nbroad](https://www.kaggle.com/datasets/nbroad/gemma-rewrite-nbroad)
- [https://www.kaggle.com/datasets/ilanmeissonnier/chatgpt-rewrite-promts/data](https://www.kaggle.com/datasets/ilanmeissonnier/chatgpt-rewrite-promts/data)
- [https://www.kaggle.com/datasets/winddude/70k-prompt-rewrite-triples/data](https://www.kaggle.com/datasets/winddude/70k-prompt-rewrite-triples/data)
- [https://www.kaggle.com/code/aatiffraz/generating-data-through-gemma7b-1000-texts-5h](https://www.kaggle.com/code/aatiffraz/generating-data-through-gemma7b-1000-texts-5h)
- ...

#### Data Preprocessing
We found that the datasets contained some useless information, and some of the useful information was noisy, so we mainly followed a kaggle expert's preprocessing approach to **denoise the data with regular expressions** (sorry, I can't find where that write-up went, so I'll just let Claude write it up😋)

The denoising covers the following:
1. Removing useless or irrelevant text:
   - The generated `rewritten_text` contains some useless or irrelevant text, such as "therefore.*I cannot", "does not contain any", etc.
   - This text is generated by the language model, but it is not the rewritten text we want.
   - By matching these patterns with regular expressions and clearing the matched text, we can remove this useless or irrelevant text.

2. Extracting the relevant text segment:
   - In some cases, `rewritten_text` contains fixed patterns such as "Sure, here.*?:", "Summary of .*?\n\n", etc.
   - These patterns are usually followed by the rewritten text we are actually interested in.
   - By matching these sub-patterns with regular expressions and extracting the text starting from the end of the match, we can get the relevant text segment.

3. Removing redundant prefixes or suffixes:
   - Sometimes the generated `rewritten_text` contains redundant prefixes or suffixes, such as text starting with "**\n\n" or "user".
   - These prefixes or suffixes come from quirks of different language models, but they are redundant for the rewritten text.
   - By stripping these redundant prefixes or suffixes, we get cleaner and more accurate rewritten text.

4. Handling special cases:
   - The regular expressions in the `fix_prompt` function can handle some special cases.
   - For example, text matching the "therefore.*I cannot" pattern indicates that the LLM most likely failed to produce a proper rewrite, so it is cleared.
   - Text matching the "Sure, here.*?:" pattern may indicate that the language model produced a fixed reply format, so the relevant text after it needs to be extracted.

**Reference code:**
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

#### Embedding Data
The data processed in the steps above is passed through the **sentence-t5-base** model to generate embeddings for the training and test sets

We also collected the unique prompts from the data above into a prompt file and used Meta's open-source Faiss library to turn it into prompt.index, which we use for similarity matching against the deberta model's predictions to produce the output **(As for why we use this method, I'll only mention one point here and explain the rest in detail later ----- this method gives a significant boost but requires a well-built private dataset. Using prompt engineering, we called gpt-4 to generate 150 high-quality mean prompts (`PB=0.58` and above, and we also found 1400000+ distinctive prompts in open-source datasets. Yes, that's 1.4 million, you read that right😀)**

**Reference code:**
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

### Training the deberta Model
**[Reference](https://www.kaggle.com/code/alejopaullier/llm-pr-seq2seq-train/notebook)**
We mainly built on this notebook and made some improvements and modifications
#### Model Configuration
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

#### Model Architecture
There are two tricks in the model architecture
- 1.We directly use deberta's pretrained weights to extract features from the original text and the rewritten text. We found **no significant difference** on the test set between full-parameter training and using the pretrained weights, and during cross-validation full training was sometimes even **worse than** the pretrained weights. So we decided to use the **pretrained weights** directly and designed a **head**, letting it learn useful representations from the rich features extracted by the underlying deberta model, and then, through transformation and compression, produce embedding vectors that effectively predict the rewrite prompt. As for why the hidden layer dimension is set to 32256, simply put, it's black magic😅. If I had to explain, it's 768*42; generally, **setting the hidden layer dimension to n times the embedding dimension** gives decent results😊. Here we started trying from `n=36` and eventually found that `n=42` worked well. (My take is that it's experience: we want the model to learn **richer semantic information** from the features deberta extracts, but we also don't want it to be **overfitting**. If this feels too much like black magic, just ensembling the two models below is already enough to get a good result **(`PV=0.6573`,`PB=0.6569`, ranked 81st on the private LB, also a silver medal position)**, so just treat this seq2seq model as a bit of fun😇)
- 2.Use BatchNorm instead of LayNorm in our head (an average gain of **0.003** across multiple cross-validation runs). I think this is a trick that only holds under **four specifics**: a specific task, a specific embedding model, a specific evaluation metric, and a specific dataset (it still held when I set batch_size to 2) 🤔

Reference code:
```python
class CustomModel(nn.Module):
    def __init__(self, cfg, config_path=None, mode: str ="train", pretrained=False): 
        super().__init__()
        self.cfg = cfg
        self.mode = mode
        self.dropout = 0.2
        if config_path is None:
            self.config = AutoConfig.from_pretrained(cfg.MODEL, output_hidden_states=True) # output_hidden_states=True enables the option to output hidden states in the config, i.e., the hidden states of every layer are kept during the forward pass.
            self.config.hidden_dropout = 0.
            self.config.hidden_dropout_prob = 0. # Dropout rate
            self.config.attention_dropout = 0. # Dropout rate of the attention weights
            self.config.attention_probs_dropout_prob = 0.

        else:
            self.config = torch.load(config_path)

        if pretrained:
            self.model = AutoModel.from_pretrained(cfg.MODEL, config=self.config) #self.config: this is a config object that contains the configuration the model should have. It is used to modify the default model config, e.g. adjusting the dropout rate or other architecture-related settings.
        else:
            self.model = AutoModel(self.config)

        """
        Purpose: enables gradient checkpointing for the model to optimize memory usage, especially when handling very large models.
        Condition: gradient checkpointing is only enabled when the GRADIENT_CHECKPOINTING attribute in the config (self.cfg) is True. This is usually specified in the config file, or set before instantiating the model class.
        Method: gradient_checkpointing_enable() is a method provided by the transformers library. It lets the model save only the necessary activations during training and recompute the others when needed, thereby reducing memory consumption.
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
        self._init_weights is a custom method used to initialize the weights of the passed-in module. This method usually contains specific initialization strategies for different types of layers (e.g. linear layers, embedding layers, layer normalization, etc.).
        This initialization includes setting the initial distribution of the weights (e.g. a normal distribution), zero-initializing the biases, etc.
        """
        self._init_weights(self.head)

    # Define the _init_weights method for custom weight initialization.
    """
    This _init_weights method provides a weight initialization strategy,
    using the most suitable initialization for each type of layer. It helps the stability and convergence speed of model training.
    """
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            """
            Linear layer (nn.Linear): if module is a linear layer, the method below initializes its weights from a normal distribution with mean 0
            and standard deviation (std) self.config.initializer_range. This initializer_range is usually specified in the model config and controls the range of the initialization distribution.
            If the layer has a bias, the bias is initialized to zero.
            """
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            """
            Embedding layer (nn.Embedding): if module is an embedding layer, its weights are likewise initialized from a normal distribution.
            If the embedding layer has a padding_idx (usually used to mark padding positions in the embedding matrix), the weights at that position are explicitly set to zero. This ensures that padding embeddings have no effect on the model
            """
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            """
            Layer normalization (nn.LayerNorm): if module is a layer normalization layer, the method initializes its bias to zero and its weight to 1.
            The weight and bias of layer normalization are usually used to adjust the scale and shift of the normalized data.
            """
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def feature(self, inputs):
        """
        self.model(**inputs): calls the pretrained Transformer model; the inputs dict contains key-value pairs such as input_ids, attention_mask, etc.
        **inputs is Python syntax that unpacks a dict into keyword arguments.
        outputs: the model output is usually an object containing multiple components. For many Hugging Face-based Transformer models, outputs contains last_hidden_state (the hidden states of the last layer),
        hidden_states (if configured to return the states of all hidden layers), and possibly attentions (if configured to return the attention weights).
        """
        outputs = self.model(**inputs)
        #last_hidden_states = outputs[1]
        feature1 = self.pool(outputs.hidden_states[-1], inputs['attention_mask'])
        feature2 = self.pool(outputs.hidden_states[-2], inputs['attention_mask'])
        """
        torch.cat([feature1, feature2], dim=1): concatenates feature1 and feature2 along the second dimension (i.e., the feature dimension).
        This fuses information from the last two hidden layers, so the resulting feature vector contains not only the contextual information of the final layer but also the semantic features of the layer before it.
        The result is an extended feature vector whose size is twice the feature size of a single layer, since it contains the outputs of two layers.
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

#### Model Training
Training the above deberta model on a 4090 on Autodl took **about 6 hours** per **epoch**

![image](https://github.com/RoschildRui/RoschildRui.github.io/assets/146306438/c10000e5-7c73-4b27-8d28-fdabf457d8c7)

We found that `batch_size=2` with GRADIENT_CHECKPOINTING off is faster than `batch_size=16` with GRADIENT_CHECKPOINTING on, and it has **no significant impact on the test set evaluation** (sometimes it even improved), so we chose not to turn on checkpointing

#### Model Inference
We match the model's predictions against the embedding index built earlier and fill in the prompt with the **highest similarity** as the prediction

Reference code:
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

Alright, finally done writing up the first model
![image](https://github.com/RoschildRui/RoschildRui.github.io/assets/146306438/509cd940-f1b4-4e54-a19b-5a010aa0a38e)

### Fine-tuning [phi](https://www.kaggle.com/models/Microsoft/phi/Transformers/2/1)
The idea comes from [Notebook1](https://www.kaggle.com/code/mozhiwenmzw/0-61-llmpr-phi2-sft-model-generate-infer/notebook) and [Notebook2](https://www.kaggle.com/code/mozhiwenmzw/0-61-llmpr-phi2-sft-model-training/notebook) open-sourced by this expert

Thanks also to this expert for open-sourcing the [Mean Prompt](https://www.kaggle.com/code/seifachour12/lb-score-0-63)

#### Training the Adapter
After going through the expert's notebooks, we first tried training phi's adapter layer on our own private dataset to make it better suited to this task

Reference code:
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
#### Fine-tuning phi with the Adapter
However, we found that whether we trained the adapter on the top layers or the middle layers of phi, we couldn't seem to match the expert's [open-source version](https://www.kaggle.com/models/mozhiwenmzw/phi2-public-data-sft-adapter/PyTorch/public-data-sft/1) (our single model reached at most `PB=0.63`, but in the ensemble the PB dropped by about 0.1 compared to using the open-source adapter)😅

So in the end we directly used the open-source adapter to fine-tune phi

We made some adjustments to the open-source code to optimize it for our final ensemble
- Based on our observations of the generated text, we added the symbols '.',';',':','<endoftext>' (sorry, the rendered format seems a bit off😳) as stop tokens for generation, to strictly control how long phi2 spends generating text
- Remove the trailing symbol from the generated text. We found that when ensembling predictions, the number of periods has to be strictly controlled; removing the period improved PB by about 0.01 🤠

Reference code:
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
    # Because of the ensemble, we don't use 'Please improve this text using the writing style with maintaining the original meaning but altering the tone.' here; use it for a single model
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

Feeling pretty safe now, kinda want to slack off a bit🫠

### few-shot mistral-7b Model
This was probably the most popular approach in the competition, in both public and private solutions

Likewise, thanks to the open-source [solution](https://www.kaggle.com/code/richolson/mistral-7b-prompt-recovery-version-2) shared here🥳

#### few-shot Constraints
Using the few-shot approach to get a large model to do prompt recovery comes with some strict constraints:
- 1. The model needs strong baseline capabilities / in this competition, the gap between few-shot llama2-13b and few-show mistral-7b was about `0.02` on PB and within `0.013` on PV
![image](https://github.com/RoschildRui/RoschildRui.github.io/assets/146306438/ec1cf9ea-ab7d-4bcb-b8a4-79da3c8fdc4d)
The figure above is the benchmark comparison officially provided by mistral. We believe the core capabilities this task mainly requires from an LLM are **Reasoning**, **Knowledge**, and **Comprehension**
- 2. Representative and diverse examples need to be chosen as the few-shot prompt data. These examples should cover the different possible variants and complexities of the task and clearly convey what the task requires. Fortunately, by learning from and distilling many discussions, we created diverse, workable examples and combined them with the open-source ones, improving PB by `0.01` and PV by `0.0072` (the code below won't include the examples we built ourselves, because we still have some doubts about a few points; we'll open-source them once those are resolved)😛
- 3. A suitable prompt needs to be designed so that the LLM pays more attention to extracting certain features

#### few-shot
We made 3 improvements on top of the open-source solution:
- 1.Changed `response_prefix = "Improve this text by"` to `response_prefix = "Improve this text using the writing style"` so the model focuses more on extracting writing style features
- 2.Specified `stop_tokens = ['.',':']` for the LLM output
- 3.If the generated text is shorter than 15 characters, replace it with a high-PB Mean prompt; for all prompts that weren't replaced, append `', maintaining the original meaning but altering the tone.'` to the end

Reference code:
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

### Ensembling the Three Models
We concatenate the predictions of the three models to get the final result🤗
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

### Summary
**Solution Summary**

While reflecting on why ensembling the predictions of the three models works well, I took 30 samples from our private dataset and ran predictions with each of the 3 models (their `PB` scores are very close: `deberta` has `PB=0.64`, `phi` has `PB=0.65`, `mistral` has `PB=0.65`). Their predictions all show some clear differences, meaning certain features of the sentences they generate are orthogonal. So my guess is that after being encoded by `sentence-t5`, these features pull the embedding closer to the target prompt, which raises the `PB` score.

After the competition ended, I saw this figure in the 1st place solution write-up
![image](https://github.com/RoschildRui/RoschildRui.github.io/assets/146306438/e4d2d756-cf67-4fd8-8b97-faafaba2d251)
Looking at this figure, I think my understanding of the ensemble is explainable to some extent😎

**What the Competition Means**
- 1.I think prompt recovery is essentially about using a model's outputs and inputs to find the most efficient way to prompt it. To me, this guides us to gradually discover how to make better use of large models in the coming era to **efficiently empower every industry**, rather than crudely piling up compute.
- 2.I feel that the way our brains work actually has some similarity to large models. Searching for the most efficient prompting method is also an **exploration in education** --- how to efficiently get learners, guided by an educator's effective prompts, to go from textbook knowledge (the original text) to the output the educator expects (the rewritten text).

### Final Words
Those of you who read all the way through probably noticed that our models in this competition were built almost entirely on **open-source** model code. The only real difference is that in the end we chose to directly concatenate the predictions of the different models, rather than doing a typical ensemble.

I think this may be what makes kaggle **so captivating** as a platform.

**kaggle** provides a platform where friends around the world who love data science and machine learning can meet. Here, **newcomers or veterans**, **beginners or experts**, everyone **exchanges ideas freely**, so every kaggler can flexibly apply what they've learned to cutting-edge scenarios, deepen their understanding through constant discussion, and learn many new things.

Thanks again to **kaggle** and **all the friends on kaggle who share so enthusiastically**!!!😉

Finally, thanks to my parents and my school for providing me with excellent conditions for study and research, and thanks to yuxuanwu and zeyu zhang for leading me onto the kaggle path












