# Multilingual Joint Fine-tuning of Transformer models for identifying Trolling, Aggression and Cyberbullying at TRAC 2020

Models and predictions for submission to TRAC - 2020 Second Workshop on Trolling, Aggression and Cyberbullying.

Our trained models as well as evaluation metrics during traing are available at: https://databank.illinois.edu/datasets/IDB-8882752#
We also make a few of our models available in HuggingFace's models repository at https://huggingface.co/socialmediaie/, these models can be further fine-tuned on your dataset of choice.

Our approach is described in our paper titled: 

> Mishra, Sudhanshu, Shivangi Prasad, and Shubhanshu Mishra. 2020. "Multilingual Joint Fine-Tuning of Transformer Models for Identifying Trolling, Aggression and Cyberbullying at TRAC 2020." In Proceedings of the Second Workshop on Trolling, Aggression and Cyberbullying (TRAC-2020).

The source code for training this model and more details can be found on our code repository: https://github.com/socialmediaie/TRAC2020

NOTE: These models are retrained for uploading here after our submission so the evaluation measures may be slightly different from the ones reported in the paper. 

If you plan to use the dataset please cite the following resources:

* Mishra, Sudhanshu, Shivangi Prasad, and Shubhanshu Mishra. 2020. "Multilingual Joint Fine-Tuning of Transformer Models for Identifying Trolling, Aggression and Cyberbullying at TRAC 2020." In Proceedings of the Second Workshop on Trolling, Aggression and Cyberbullying (TRAC-2020).
* Mishra, Shubhanshu, Shivangi Prasad, and Shubhanshu Mishra. 2020. “Trained Models for Multilingual Joint Fine-Tuning of Transformer Models for Identifying Trolling, Aggression and Cyberbullying at TRAC 2020.” University of Illinois at Urbana-Champaign. https://doi.org/10.13012/B2IDB-8882752_V1.


```
@inproceedings{Mishra2020TRAC,
author = {Mishra, Sudhanshu and Prasad, Shivangi and Mishra, Shubhanshu},
booktitle = {Proceedings of the Second Workshop on Trolling, Aggression and Cyberbullying (TRAC-2020)},
title = {{Multilingual Joint Fine-tuning of Transformer models for identifying Trolling, Aggression and Cyberbullying at TRAC 2020}},
year = {2020}
}

@data{illinoisdatabankIDB-8882752,
author = {Mishra, Shubhanshu and Prasad, Shivangi and Mishra, Shubhanshu},
doi = {10.13012/B2IDB-8882752_V1},
publisher = {University of Illinois at Urbana-Champaign},
title = {{Trained models for Multilingual Joint Fine-tuning of Transformer models for identifying Trolling, Aggression and Cyberbullying at TRAC 2020}},
url = {https://doi.org/10.13012/B2IDB-8882752{\_}V1},
year = {2020}
}
```


## Usage

The models can be used via the following code: 

```python
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
import torch
from pathlib import Path
from scipy.special import softmax
import numpy as np
import pandas as pd

TASK_LABEL_IDS = {
    "Sub-task A": ["OAG", "NAG", "CAG"],
    "Sub-task B": ["GEN", "NGEN"],
    "Sub-task C": ["OAG-GEN", "OAG-NGEN", "NAG-GEN", "NAG-NGEN", "CAG-GEN", "CAG-NGEN"]
}

model_version="databank" # other option is hugging face library
if model_version == "databank":
    # Make sure you have downloaded the required model file from https://databank.illinois.edu/datasets/IDB-8882752
    # Unzip the file at some model_path (we are using: "databank_model")
    model_path = next(Path("databank_model").glob("./*/output/*/model"))
    # Assuming you get the following type of structure inside "databank_model"
    # 'databank_model/ALL/Sub-task C/output/bert-base-multilingual-uncased/model'
    lang, task, _, base_model, _ = model_path.parts
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
else:
    lang, task, base_model = "ALL", "Sub-task C", "bert-base-multilingual-uncased"
    base_model = f"socialmediaie/TRAC2020_{lang}_{lang.split()[-1]}_{base_model}"
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(base_model)

# For doing inference set model in eval mode
model.eval()
# If you want to further fine-tune the model you can reset the model to model.train()

task_labels = TASK_LABEL_IDS[task]

sentence = "This is a good cat and this is a bad dog."
processed_sentence = f"{tokenizer.cls_token} {sentence}"
tokens = tokenizer.tokenize(sentence)
indexed_tokens = tokenizer.convert_tokens_to_ids(tokens)
tokens_tensor = torch.tensor([indexed_tokens])

with torch.no_grad():
  logits, = model(tokens_tensor, labels=None)
logits


preds = logits.detach().cpu().numpy()
preds_probs = softmax(preds, axis=1)
preds = np.argmax(preds_probs, axis=1)
preds_labels = np.array(task_labels)[preds]
print(dict(zip(task_labels, preds_probs[0])), preds_labels)
"""You should get an output as follows:

({'CAG-GEN': 0.06762535,
  'CAG-NGEN': 0.03244293,
  'NAG-GEN': 0.6897794,
  'NAG-NGEN': 0.15498641,
  'OAG-GEN': 0.034373745,
  'OAG-NGEN': 0.020792078},
 array(['NAG-GEN'], dtype='<U8'))

"""

```