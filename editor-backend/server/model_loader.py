import torch
from torch import nn
import torch.nn.functional as F
import time
import gc
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    AutoModelForCausalLM,
    StoppingCriteriaList,
    StoppingCriteria,
)

gc.enable()
SEMANTIC_EMBEDDING = "sentence-transformers/all-mpnet-base-v2"
ROBERTA_PATH = "./models/clrp-roberta-base/clrp_roberta_base"
READABILITY_MODELS = [
    "./models/model_1.pth",
    "./models/model_2.pth",
    "./models/model_3.pth",
]
RLHF_MODEL_PATH = "./models/actor"
RLHF_MODEL_NAME = "facebook/opt-1.3b"
LLM_MODEL = "tiiuae/falcon-7b"
BATCH_SIZE = 1
MAX_LEN = 256
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
        last_layer_hidden_states = roberta_output.hidden_states[-1]
        weights = self.attention(last_layer_hidden_states)
        context_vector = torch.sum(weights * last_layer_hidden_states, dim=1)
        return self.regressor(context_vector)


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


def load_similarity_model():
    similarity_model = AutoModel.from_pretrained(SEMANTIC_EMBEDDING)
    similarity_tokenizer = AutoTokenizer.from_pretrained(SEMANTIC_EMBEDDING)
    return similarity_model, similarity_tokenizer


def _load_readability_model(model_location):
    m = LitModel()
    m.load_state_dict(torch.load(model_location))
    m.to(DEVICE)
    return m


def load_readability_models():
    models = []
    for m in READABILITY_MODELS:
        models.append(_load_readability_model(m))
    readability_tokenizer = AutoTokenizer.from_pretrained(ROBERTA_PATH)
    return models, readability_tokenizer


def load_rlhf_model():
    rlhf_model = AutoModelForCausalLM.from_pretrained(RLHF_MODEL_PATH).to(DEVICE)
    rlhf_tokenizer = AutoTokenizer.from_pretrained(RLHF_MODEL_NAME)
    return rlhf_model, rlhf_tokenizer


def load_llm_model():
    llm_model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(DEVICE)
    llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    return llm_model, llm_tokenizer


def load_model_workspace():
    read_models, read_tokenizer = load_readability_models()
    sem_model, sem_tokenizer = load_similarity_model()
    tuned_model, tuned_tokenizer = load_rlhf_model()
    large_model, large_tokenizer = load_llm_model()
    return {
        "similarity_model": sem_model,
        "similarity_tokenizer": sem_tokenizer,
        "readability_models": read_models,
        "readability_tokenizer": read_tokenizer,
        "tuned_model": tuned_model,
        "tuned_tokenizer": tuned_tokenizer,
        "large_model": large_model,
        "large_tokenizer": large_tokenizer,
    }


def mean_pooling(model_output, attention_mask):
    # Mean Pooling - Take attention mask into account for correct averaging
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def infer_similarity(
    model, tokenizer, input_prompt, output_response, kwargs=None
) -> float:
    with torch.no_grad():
        encoded_input = tokenizer(
            input_prompt, padding=True, truncation=True, return_tensors="pt"
        )
        encoded_output = tokenizer(
            output_response, padding=True, truncation=True, return_tensors="pt"
        )
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


def infer_partial_readability(model, encoded_input) -> float:
    input_ids = encoded_input["input_ids"].to(DEVICE)
    attention_mask = encoded_input["attention_mask"].to(DEVICE)

    pred = model(input_ids, attention_mask)
    result = pred.flatten().to("cpu")
    return result.numpy()[0]


def infer_single_readability(models, tokenizer, prompt, kwargs=None):
    with torch.no_grad():
        r_encoded_input = tokenizer(
            prompt, padding=True, truncation=True, return_tensors="pt"
        )
        input_readability_preds = [
            infer_partial_readability(models[0], r_encoded_input),
            infer_partial_readability(models[1], r_encoded_input),
            infer_partial_readability(models[2], r_encoded_input),
        ]
    return sum(input_readability_preds) / len(input_readability_preds)


def infer_readability(models, tokenizer, input_prompt, output_response):
    r_encoded_input = tokenizer(
        input_prompt, padding=True, truncation=True, return_tensors="pt"
    )
    r_encoded_output = tokenizer(
        output_response, padding=True, truncation=True, return_tensors="pt"
    )
    input_readability_preds = [
        infer_partial_readability(models[0], r_encoded_input),
        infer_partial_readability(models[1], r_encoded_input),
        infer_partial_readability(models[2], r_encoded_input),
    ]
    output_readability_preds = [
        infer_partial_readability(models[0], r_encoded_output),
        infer_partial_readability(models[1], r_encoded_output),
        infer_partial_readability(models[2], r_encoded_output),
    ]
    in_readability = sum(input_readability_preds) / len(input_readability_preds)
    out_readability = sum(output_readability_preds) / len(output_readability_preds)
    diff_readability = out_readability - in_readability
    return in_readability, out_readability, diff_readability


def wrap_rewrite_request(prompt):
    prepend = (
        "Rewrite the following passage to be more readable to lower grade level readers."
        "###\n"
        'Passage: "When the young people returned to the ballroom, it presented a decidedly '
        'changed appearance." Rewrite: "When the young people returned to the ballroom, it had '
        'a different appearance."'
        "###\n"
        'Passage: "This Pedrarias was seventy-two years old."'
        'Rewrite: "This jewel was seventy-two years old."'
        "###\n"
        "Passage: "
    )
    return f'{prepend}"{prompt}." Rewrite: "'


def generate_rewrite_rlhf(model, tokenizer, prompt, kwargs):
    query = wrap_rewrite_request(prompt)
    s_model, s_tokenizer, r_models, r_tokenizer = (
        kwargs["similarity_model"],
        kwargs["similarity_tokenizer"],
        kwargs["readability_models"],
        kwargs["readability_tokenizer"],
    )
    input_ids = tokenizer(query, return_tensors="pt").input_ids.to(DEVICE)
    text_ids = tokenizer(prompt, return_tensors="pt").input_ids
    in_len = int(len(text_ids[0]) * 1.2)
    attention_mask = tokenizer(query, return_tensors="pt").attention_mask.to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=in_len,
            early_stopping=True,
            do_sample=True,
        )
        answer = tokenizer.batch_decode(
            out, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        rewrite = answer[0][answer[0].find(query) + len(query) :]
        rewrite = rewrite.split("#")[0]
        rewrite = rewrite.split("<|")[0]
        in_readability, out_readability, readability_diff = infer_readability(
            r_models, r_tokenizer, prompt, rewrite
        )
        return {
            "prompt": query,
            "inference": answer[0],
            "input": prompt,
            "response": rewrite,
            "original_readability": f"{in_readability:.2f}",
            "rewrite_readability": f"{out_readability:.2f}",
            "readability_difference": f"{readability_diff:.2f}",
            "rewrite_similarity": f"{infer_similarity(s_model, s_tokenizer, prompt, rewrite):.2f}",
        }


def generate_rewrite_falcon(model, tokenizer, prompt, kwargs):
    query = wrap_rewrite_request(prompt)
    s_model, s_tokenizer, r_models, r_tokenizer = (
        kwargs["similarity_model"],
        kwargs["similarity_tokenizer"],
        kwargs["readability_models"],
        kwargs["readability_tokenizer"],
    )
    input_ids = tokenizer(query, return_tensors="pt").input_ids.to(DEVICE)
    text_ids = tokenizer(prompt, return_tensors="pt").input_ids
    in_len = int(len(text_ids[0]) * 1.2)
    attention_mask = tokenizer(query, return_tensors="pt").attention_mask.to(DEVICE)
    stop_words_ids = [
        tokenizer(stop_word, return_tensors="pt")["input_ids"].squeeze()
        for stop_word in ["###"]
    ]
    stopping_criteria = StoppingCriteriaList(
        [StoppingCriteriaSub(stops=stop_words_ids, encounters=4)]
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
        answer = tokenizer.batch_decode(
            out, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        rewrite = answer[0][answer[0].find(query) + len(query) :]
        rewrite = rewrite.split("#")[0]
        in_readability, out_readability, readability_diff = infer_readability(
            r_models, r_tokenizer, prompt, rewrite
        )
        return {
            "prompt": query,
            "inference": answer[0],
            "input": prompt,
            "response": rewrite,
            "original_readability": f"{in_readability:.2f}",
            "rewrite_readability": f"{out_readability:.2f}",
            "readability_difference": f"{readability_diff:.2f}",
            "rewrite_similarity": f"{infer_similarity(s_model, s_tokenizer, prompt, rewrite):.2f}",
        }


def time_method_call(method, args, kwargs):
    start = time.time()
    print(method(*args, kwargs=kwargs))
    end = time.time()
    print(f"Method call time: {end - start}")


if __name__ == "__main__":
    gc.collect()
    test_prompt = (
        "King Edward, be it remembered, was a man of many and varied interests"
    )
    workspace = load_model_workspace()

    readability = [
        workspace["readability_models"],
        workspace["readability_tokenizer"],
        test_prompt,
    ]
    similarity = [
        workspace["similarity_model"],
        workspace["similarity_tokenizer"],
        test_prompt,
        test_prompt,
    ]
    tuned = [workspace["tuned_model"], workspace["tuned_tokenizer"], test_prompt]
    large = [workspace["large_model"], workspace["large_tokenizer"], test_prompt]

    gc.collect()
    time_method_call(infer_single_readability, readability, kwargs=workspace)

    gc.collect()
    time_method_call(infer_similarity, similarity, kwargs=workspace)

    gc.collect()
    time_method_call(generate_rewrite_rlhf, tuned, kwargs=workspace)

    gc.collect()
    time_method_call(generate_rewrite_falcon, large, kwargs=workspace)
