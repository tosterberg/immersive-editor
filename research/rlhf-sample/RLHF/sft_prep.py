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


SEED = 1000
BATCH_SIZE = 1
MAX_LEN = 248
ROBERTA_PATH = "./models/clrp-roberta-base/clrp_roberta_base"
READABILITY_MODELS = [
    "./models/model_1.pth",
    "./models/model_2.pth",
    "./models/model_3.pth",
]
SEMANTIC_EMBEDDING = "sentence-transformers/all-mpnet-base-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class LitModel(nn.Module):
    def __init__(self):
        super().__init__()

        config = AutoConfig.from_pretrained(ROBERTA_PATH)
        config.update(
            {
                "output_hidden_states": True,
                "hidden_dropout_prob": 0.0,
                "layer_norm_eps": 1e-7,
            }
        )

        self.roberta = AutoModel.from_pretrained(ROBERTA_PATH, config=config)

        self.attention = nn.Sequential(
            nn.Linear(768, 512), nn.Tanh(), nn.Linear(512, 1), nn.Softmax(dim=1)
        )

        self.regressor = nn.Sequential(nn.Linear(768, 1))

    def forward(self, input_ids, attention_mask):
        roberta_output = self.roberta(
            input_ids=input_ids, attention_mask=attention_mask
        )

        # There are a total of 13 layers of hidden states.
        # 1 for the embedding layer, and 12 for the 12 Roberta layers.
        # We take the hidden states from the last Roberta layer.
        last_layer_hidden_states = roberta_output.hidden_states[-1]

        # The number of cells is MAX_LEN.
        # The size of the hidden state of each cell is 768 (for roberta-base).
        # In order to condense hidden states of all cells to a context vector,
        # we compute a weighted average of the hidden states of all cells.
        # We compute the weight of each cell, using the attention neural network.
        weights = self.attention(last_layer_hidden_states)

        # weights.shape is BATCH_SIZE x MAX_LEN x 1
        # last_layer_hidden_states.shape is BATCH_SIZE x MAX_LEN x 768
        # Now we compute context_vector as the weighted average.
        # context_vector.shape is BATCH_SIZE x 768
        context_vector = torch.sum(weights * last_layer_hidden_states, dim=1)

        # Now we reduce the context vector to the prediction score.
        return self.regressor(context_vector)


def mean_pooling(model_output, attention_mask):
    # Mean Pooling - Take attention mask into account for correct averaging
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


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


def infer_similarity(model, encoded_input, encoded_output) -> float:
    sim_output = model(**encoded_input)
    sentence_embeddings_input = mean_pooling(
        sim_output, encoded_input["attention_mask"]
    )
    sim_output = model(**encoded_output)
    sentence_embeddings_output = mean_pooling(
        sim_output, encoded_output["attention_mask"]
    )
    # Normalize embeddings
    sentence_embeddings_input = F.normalize(sentence_embeddings_input, p=2, dim=1)
    sentence_embeddings_output = F.normalize(sentence_embeddings_output, p=2, dim=1)
    return torch.cosine_similarity(
        sentence_embeddings_input, sentence_embeddings_output
    ).numpy()[0]


def infer_readability(model, encoded_input) -> float:
    input_ids = encoded_input["input_ids"].to(DEVICE)
    attention_mask = encoded_input["attention_mask"].to(DEVICE)

    pred = model(input_ids, attention_mask)
    result = pred.flatten().to("cpu")
    return result.numpy()[0]


def set_random_seed(random_seed):
    """ Coordinates random seeds for various packages for deterministic results """
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ["PYTHONHASHSEED"] = str(random_seed)

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    torch.device(DEVICE)
    set_random_seed(SEED)

    # sentences = make_new_dataset(train_df)
    # with open("sentence_dataset.json", 'w') as f:
    #     json.dump(sentences, f)
    response_list = []
    with open("data/sentence_second_inferences.json", "r") as f:
        sentence_dataset = json.load(f)

    similarity_model = AutoModel.from_pretrained(SEMANTIC_EMBEDDING)
    similarity_tokenizer = AutoTokenizer.from_pretrained(SEMANTIC_EMBEDDING)

    r1 = LitModel()
    r1.load_state_dict(torch.load(READABILITY_MODELS[0]))
    r1.to(DEVICE)

    r2 = LitModel()
    r2.load_state_dict(torch.load(READABILITY_MODELS[1]))
    r2.to(DEVICE)

    r3 = LitModel()
    r3.load_state_dict(torch.load(READABILITY_MODELS[2]))
    r3.to(DEVICE)

    readability_tokenizer = AutoTokenizer.from_pretrained(ROBERTA_PATH)

    dataset = SentenceInferenceDataset(sentence_dataset["dataset"])
    data = dataset.get_all_data()
    acceptance_count = 0

    for idx, obs in enumerate(data):
        if idx % 1000 == 0:
            print(
                f"Inference {idx}/{len(data)}: Evaluation completion {idx/len(data)*100:.2f}%"
            )
        ins = obs["input"]
        outs = obs["response"]
        outs.replace("#", "")
        sim_encoded_input = similarity_tokenizer(
            ins, padding=True, truncation=True, return_tensors="pt"
        )
        sim_encoded_output = similarity_tokenizer(
            outs, padding=True, truncation=True, return_tensors="pt"
        )
        r_encoded_input = readability_tokenizer(
            ins, padding=True, truncation=True, return_tensors="pt"
        )
        r_encoded_output = readability_tokenizer(
            outs, padding=True, truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            similarity = infer_similarity(
                similarity_model, sim_encoded_input, sim_encoded_output
            )
            input_readability_preds = [
                infer_readability(r1, r_encoded_input),
                infer_readability(r2, r_encoded_input),
                infer_readability(r3, r_encoded_input),
            ]
            output_readability_preds = [
                infer_readability(r1, r_encoded_output),
                infer_readability(r2, r_encoded_output),
                infer_readability(r3, r_encoded_output),
            ]
            in_readability = sum(input_readability_preds) / len(input_readability_preds)
            out_readability = sum(output_readability_preds) / len(
                output_readability_preds
            )
            diff_readability = out_readability - in_readability
            feedback = diff_readability >= 0 and similarity >= 0.75
            if feedback:
                acceptance_count += 1
            response_list.append(
                {
                    idx: {
                        "prompt": obs["prompt"],
                        "inference": obs["inference"],
                        "input": obs["input"],
                        "response": obs["response"],
                        "readability_input": str(in_readability),
                        "readability_output": str(out_readability),
                        "readability_diff": str(diff_readability),
                        "similarity": str(similarity),
                        "accepted": str(feedback),
                    }
                }
            )
    response_dataset = {"dataset": response_list}
    print(
        f"Done... Acceptance {acceptance_count/len(data) * 100:.2f}%, Total {acceptance_count}"
    )
    # with open("data/sentence_sft_dataset.json", "w") as f:
    with open("data/sentence_sft_second_dataset.json", "w") as f:
        json.dump(response_dataset, f)
"""
"""
