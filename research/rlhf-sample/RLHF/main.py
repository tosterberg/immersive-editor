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
    AutoModelForCausalLM,
    pipeline,
    StoppingCriteriaList,
    StoppingCriteria,
)

SEED = 1234
BATCH_SIZE = 1
MAX_LEN = 248
LLM_PATH = "tiiuae/falcon-7b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SentenceDataset(Dataset):
    def __init__(self, prompts):
        super().__init__()

        self.prompts = prompts
        self.prepend = (
            "Rewrite the following passage to be more readable to lower grade level readers."
            "###\n"
            'Passage: "This Pedrarias was seventy-two years old."'
            'Rewrite: "This jewel or stone was seventy-two years old."'
            "###\n"
            "Passage: "
        )

        # self.prepend = (
        #    "Rewrite the following passage to be more readable to lower grade level readers."
        #    "###\n"
        #    'Passage: "When the young people returned to the ballroom, it presented a decidedly '
        #    'changed appearance." Rewrite: "When the young people returned to the ballroom, it had '
        #    'a different appearance."'
        #    "###\n"
        #    'Passage: "This Pedrarias was seventy-two years old."'
        #    'Rewrite: "This jewel was seventy-two years old."'
        #    "###\n"
        #    "Passage: "
        # )

        self.prepended_prompts = [
            f'{self.prepend}"{prompt}." Rewrite: "' for prompt in self.prompts
        ]

    def __len__(self):
        return len(self.prepended_prompts)

    def all_prompts(self):
        return self.prepended_prompts, self.prompts

    def __getitem__(self, index):
        return None, None


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops
        self.ENCOUNTERS = encounters

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        stop_count = 0
        for stop in self.stops:
            stop_count = (stop == input_ids[0]).sum().item()

        if stop_count >= self.ENCOUNTERS:
            return True
        return False


def set_random_seed(random_seed):
    """ Coordinates random seeds for various packages for deterministic results """
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ["PYTHONHASHSEED"] = str(random_seed)

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    torch.backends.cudnn.deterministic = True


def make_new_dataset(df):
    texts = df.excerpt.tolist()
    sentences = []
    for text in texts:
        text.replace("?", ".")
        text.replace("!", ".")
        samples = text.split(". ")
        for sample in samples:
            sentences.append(sample)
    return {"dataset": sentences}


if __name__ == "__main__":
    torch.device(DEVICE)
    set_random_seed(SEED)

    # sentences = make_new_dataset(train_df)
    # with open("sentence_dataset.json", 'w') as f:
    #     json.dump(sentences, f)
    response_list = []
    with open("data/archive/sentence_dataset.json", "r") as f:
        sentence_list = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(LLM_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(DEVICE)
    dataset = SentenceDataset(sentence_list["dataset"])
    stop_words_ids = [
        tokenizer(stop_word, return_tensors="pt")["input_ids"].squeeze()
        for stop_word in ["###"]
    ]
    stopping_criteria = StoppingCriteriaList(
        [StoppingCriteriaSub(stops=stop_words_ids, encounters=4)]
    )
    prompts, texts = dataset.all_prompts()

    for idx, prompt in enumerate(prompts):
        if idx % 100 == 0:
            print(f"{idx}/{len(prompts)}: Inference\n")
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
        text_ids = tokenizer(texts[idx], return_tensors="pt").input_ids
        in_len = int(len(text_ids[0]) * 1.2)
        attention_mask = tokenizer(prompt, return_tensors="pt").attention_mask.to(
            DEVICE
        )
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=in_len,
                early_stopping=True,
                stopping_criteria=stopping_criteria,
                do_sample=True,
                pad_token_id=11,
            )
            answer = tokenizer.batch_decode(out, skip_special_tokens=True)
            response_list.append(
                {
                    idx: {
                        "prompt": prompt,
                        "inference": answer[0],
                        "input": texts[idx],
                        "response": answer[0][answer[0].find(prompt) + len(prompt) :],
                    }
                }
            )
    print("Done")
    response_dataset = {"dataset": response_list}
    # with open("data/sentence_inferences.json", "w") as f:
    with open("data/sentence_second_inferences.json", "w") as f:
        json.dump(response_dataset, f)
"""
"""
