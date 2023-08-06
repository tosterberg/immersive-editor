import torch
import random
import numpy as np
import os
import einops
import json
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    AutoModelForCausalLM,
    pipeline,
    StoppingCriteriaList,
    StoppingCriteria,
)
import torch
from torch import nn
import torch.nn.functional as F


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SentenceInferenceDataset(Dataset):
    def __init__(self, dataset_json):
        super().__init__()
        self.dataset = dataset_json
        self.pairs = []
        self.observations = []
        for observation in self.dataset:
            for val in observation.values():
                self.observations.append(val)
                self.pairs.append([val["input"], val["response"]])

    def __len__(self):
        return len(self.dataset)

    def get_pair_sentences(self):
        return self.pairs

    def get_all_data(self):
        return self.observations

    def __getitem__(self, index):
        return None, None


def decompose_responses(res1, res2):
    if res1["accepted"] == "True" and res2["accepted"] != "True":
        return res1["prompt"], res1["response"], res1["response"], res2["response"]
    if res2["accepted"] == "True" and res1["accepted"] != "True":
        return res2["prompt"], res2["response"], res2["response"], res1["response"]
    if res1["accepted"] == "True" and res2["accepted"] == "True":
        m1 = float(res1["readability_diff"]) - (1.0 - float(res1["similarity"]))
        m2 = float(res2["readability_diff"]) - (1.0 - float(res2["similarity"]))
        if m1 >= m2:
            return res1["prompt"], res1["response"], res1["response"], res2["response"]
    return res2["prompt"], res2["response"], res2["response"], res1["response"]


if __name__ == "__main__":
    torch.device(DEVICE)

    prompts = []
    responses = []
    chosens = []
    rejecteds = []
    eval_prompts = []
    eval_responses = []
    eval_chosens = []
    eval_rejecteds = []
    with open("data/sentence_sft_dataset.json", "r") as f:
        sentence_dataset_accept = json.load(f)
    with open("data/sentence_sft_second_dataset.json", "r") as f:
        sentence_dataset_reject = json.load(f)

    accept_dataset = SentenceInferenceDataset(sentence_dataset_accept["dataset"])
    reject_dataset = SentenceInferenceDataset(sentence_dataset_reject["dataset"])
    accept_data = accept_dataset.get_all_data()
    reject_data = reject_dataset.get_all_data()

    if len(accept_data) != len(reject_data):
        raise AssertionError("Accept Dataset must equal Reject Dataset")

    count = 0
    for idx in range(len(accept_data)):
        a_obs = accept_data[idx]
        r_obs = reject_data[idx]
        if a_obs["accepted"] == "True" or r_obs["accepted"] == "True":
            prompt, response, accept, reject = decompose_responses(a_obs, r_obs)
            count += 1
            if count % 5 == 0:
                eval_prompts.append(prompt)
                eval_responses.append(response)
                eval_chosens.append(accept)
                eval_rejecteds.append(reject)
            else:
                prompts.append(prompt)
                responses.append(response)
                chosens.append(accept)
                rejecteds.append(reject)
    train = {"prompt": prompts, "chosen": chosens, "rejected": rejecteds}
    test = {"prompt": eval_prompts, "chosen": eval_chosens, "rejected": eval_rejecteds}
    with open("dataset/train.json", "w") as f:
        json.dump(train, f)
    with open("dataset/eval.json", "w") as f:
        json.dump(test, f)
"""
"""
