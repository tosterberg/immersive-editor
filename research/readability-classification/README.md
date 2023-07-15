# Classification of Sentence Readability

## Methodology
Our goal is to train a model that can predict the reading level of a sentence, which will be used for our editor to highlight sentences that will be difficult to understand. For the purposes of our design we will be returning sentences that score in the top quartile of reading level as sentences to be rewritten. With inference being the final product of this training we are going to look at minimizing the size, number, and latency of the models used. The Kaggle leaderboard shows a top score of 0.446 RMSE, so if we can train something around 0.50 RMSE I will call that sufficient for this POC.

We will use the CommonLit Readability Prize dataset from Kaggle, as our initial dataset, and our strategy will be to finetune a pretrained model for the task. The architecture selected for training is the roberta-base, which we will use a version that is pretrained on the CLRP dataset.

## Dataset
[CommonLit Readability Prize](https://www.kaggle.com/competitions/commonlitreadabilityprize/data)

## References
[Pretrained Base Model - CLRP Roberta Base](https://www.kaggle.com/datasets/maunish/clrp-roberta-base)
