# Models

## Overview
The models that I have used to make this project work locally can be obtained in their untrained, and pretrained states from Kaggle and HuggingFace. I will not be putting the weights up in this repository, but the workflow should be usable with some tweaking in order to reproduce the dataset, training data, and rlhf model.

## Specifics

Sentence Similarity
- [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)

Readability 
- [Pretrained Base Model - CLRP Roberta Base](https://www.kaggle.com/datasets/maunish/clrp-roberta-base)
- Trained on [Kaggle dataset](https://www.kaggle.com/competitions/commonlitreadabilityprize)
- Inference is run with 3 fine-tuned Roberta-Base models outputs averaged together

Sentence Rewrites
- "Human Feedback" faking model is a few-shot [falcon-7b](https://huggingface.co/tiiuae/falcon-7b)
- Fine-tuning and Actor model is [opt-1.3b](https://huggingface.co/facebook/opt-1.3b)
- Reward and Critic model is [opt-350m](https://huggingface.co/facebook/opt-350m)