import torch
from torch import nn
import torch.nn.functional as F
import json
import gc
import os
import uuid
import copy
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    AutoModelForCausalLM,
    StoppingCriteriaList,
    StoppingCriteria,
)


gc.enable()
cwd = os.getcwd()
FEEDBACK_RESPONSES = "feedback/responses.json"
FEEDBACK_REWRITES = "feedback/rewrites.json"
FEEDBACK_PROMPTS = "feedback/prompts.json"
SEMANTIC_EMBEDDING = "sentence-transformers/all-mpnet-base-v2"
ROBERTA_PATH = os.path.join(cwd, "models/clrp-roberta-base/clrp_roberta_base")
READABILITY_MODELS = [
    "models/model_1.pth",
    "models/model_2.pth",
    "models/model_3.pth",
]
RLHF_MODEL_PATH = "models/actor"
RLHF_MODEL_NAME = "facebook/opt-1.3b"
LLM_MODEL = "tiiuae/falcon-7b"
BATCH_SIZE = 1
MAX_LEN = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class LitModel(nn.Module):
    """
        LitModel

        Module implementation of the custom base models used for predicting the readability of a sentence.
    """

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
    """
        StoppingCriteriaSub

        Customer stopping criteria for the falcon-7b model to produce better output.
    """

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


class ReadabilityWorkflow(object):
    """
        ReadabilityWorkflow

        Object for orchestrating the whole readability rewrite workflow, as well as,
        loading and logging for useful persisted data created while running the predict method.
    """

    def __init__(
        self,
        llm=False,
        annotations_path=FEEDBACK_RESPONSES,
        rewrites_path=FEEDBACK_REWRITES,
        prompts_path=FEEDBACK_PROMPTS,
    ):
        self.similarity_model = SimilarityModelWrapper()
        self.readability_model = ReadabilityModelWrapper()
        self.rlhf_model = RLHFModelWrapper()
        if llm:
            self.large_model = LargeModelWrapper()
        else:
            self.large_model = None
        self.annotations_path = annotations_path
        self.rewrites_path = rewrites_path
        self.prompts_path = prompts_path
        self.prompts = {}
        self.responses = []
        self.rated_responses = []
        self.load_annotations()

    def save_annotations(self):
        annotations_file = open(self.annotations_path, "w")
        json.dump(self.rated_responses, annotations_file)
        annotations_file.close()

        rewrites_file = open(self.rewrites_path, "w")
        json.dump(self.responses, rewrites_file)
        rewrites_file.close()

        prompts_file = open(self.prompts_path, "w")
        json.dump(self.prompts, prompts_file)
        prompts_file.close()

    def load_annotations(self):
        annotations_file = open(self.annotations_path, "r")
        self.rated_responses = json.load(annotations_file)
        annotations_file.close()

        rewrites_file = open(self.rewrites_path, "r")
        self.responses = json.load(rewrites_file)
        rewrites_file.close()

        prompts_file = open(self.prompts_path, "r")
        self.prompts = json.load(prompts_file)
        prompts_file.close()

    def add_rated_response(self, annotations):
        for annotation in annotations:
            self.rated_responses.append(annotation)

    def get_prompt_id(self, prompt):
        if prompt not in self.prompts.keys():
            self.prompts[prompt] = str(uuid.uuid4())
        return self.prompts[prompt]

    def build_response(self, responses, count):
        responses.sort(key=lambda x: x["quality_score"], reverse=True)
        for response in responses:
            self.responses.append(response)
        return dict({"inferences": responses[:count]})

    def generate_rewrite(self, predictor, config, responses, retries=10):
        responses_adjusted_retries = responses * retries
        response_list = []
        for attempt in range(responses_adjusted_retries):
            config["query"], config["response"], config["rewrite"] = predictor.predict(
                config["prompt"]
            )
            config["rewrite_readability"] = float(
                self.readability_model.predict(config["rewrite"])
            )
            config["similarity"] = float(
                self.similarity_model.predict(config["prompt"], config["rewrite"])
            )
            config["model_name"] = predictor.name
            delta_readability = (
                config["rewrite_readability"] - config["original_readability"]
            )
            delta_similarity = (1 - abs(0.92 - config["similarity"])) ** 2
            config["readability_change"] = delta_readability
            config["quality_score"] = (delta_readability + 1) * delta_similarity - 1
            config["rewrite_id"] = str(uuid.uuid4())
            if delta_readability > 0.0 and config["similarity"] > 0.8:
                response_list.append(copy.deepcopy(config))
            if len(response_list) >= responses:
                break
        return response_list

    def predict(self, prompt, responses=4):
        prompt_id = self.get_prompt_id(prompt)
        original_readability = self.readability_model.predict(prompt)
        config = {
            "prompt_id": prompt_id,
            "prompt": prompt,
            "original_readability": original_readability,
        }
        gc.collect()
        rlhf = copy.deepcopy(config)
        rlhf = self.generate_rewrite(self.rlhf_model, rlhf, responses)
        gc.collect()
        if self.large_model:
            llm = copy.deepcopy(config)
            llm = self.generate_rewrite(self.large_model, llm, responses)
            gc.collect()
            return self.build_response((rlhf + llm), responses)
        return self.build_response(rlhf, responses)


class LargeModelWrapper(object):
    """
        LargeModelWrapper
    
        Generates a rewrite of a sentence which is meant to be easier to read, with a predict method
        that returns the context wrapped prompt, the full response, and a stripped sentence that is
        the rewrite for the prompt.
    """

    def __init__(
        self,
        model_id_or_path=LLM_MODEL,
        device=DEVICE,
        dtype=torch.bfloat16,
        name="flacon-7b",
    ):
        self.model_id_or_path = model_id_or_path
        self.device = device
        self.model = None
        self.tokenizer = None
        self.dtype = dtype
        self.name = name
        self._load()

    def _load(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id_or_path, trust_remote_code=True, torch_dtype=self.dtype
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id_or_path)

    @staticmethod
    def _context_wrap(prompt):
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

    def predict(self, prompt):
        query = self._context_wrap(prompt)
        input_ids = self.tokenizer(query, return_tensors="pt").input_ids.to(DEVICE)
        text_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        in_len = int(len(text_ids[0]) * 1.2)
        attention_mask = self.tokenizer(query, return_tensors="pt").attention_mask.to(
            DEVICE
        )
        stop_words_ids = [
            self.tokenizer(stop_word, return_tensors="pt")["input_ids"].squeeze()
            for stop_word in ["###"]
        ]
        stopping_criteria = StoppingCriteriaList(
            [StoppingCriteriaSub(stops=stop_words_ids, encounters=4)]
        )
        with torch.no_grad():
            out = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=in_len,
                early_stopping=True,
                stopping_criteria=stopping_criteria,
                do_sample=True,
                pad_token_id=11,
            )
            answer = self.tokenizer.batch_decode(
                out, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            rewrite = answer[0][answer[0].find(query) + len(query) :]
            rewrite = rewrite.split('."')[0]
            rewrite = rewrite.split("#")[0]
            rewrite.strip()
            return query, answer[0], rewrite


class RLHFModelWrapper(object):
    """
        RLHFModelWrapper

        Generates a rewrite of a sentence which is meant to be easier to read, with a predict method
        that returns the context wrapped prompt, the full response, and a stripped sentence that is
        the rewrite for the prompt.
    """

    def __init__(
        self,
        model_id_or_path=RLHF_MODEL_PATH,
        tokenizer_name=RLHF_MODEL_NAME,
        device=DEVICE,
        name="rlhf-opt-1.3b",
    ):
        self.model_id_or_path = model_id_or_path
        self.tokenizer_name = tokenizer_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self.name = name
        self._load()

    def _load(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id_or_path).to(
            self.device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

    @staticmethod
    def _context_wrap(prompt):
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

    def predict(self, prompt):
        query = self._context_wrap(prompt)
        input_ids = self.tokenizer(query, return_tensors="pt").input_ids.to(self.device)
        text_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        in_len = int(len(text_ids[0]) * 1.2)
        attention_mask = self.tokenizer(query, return_tensors="pt").attention_mask.to(
            self.device
        )
        with torch.no_grad():
            out = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=in_len,
                early_stopping=True,
                do_sample=True,
            )
            answer = self.tokenizer.batch_decode(
                out, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            rewrite = answer[0][answer[0].find(query) + len(query) :]
            rewrite = rewrite.split('."')[0]
            rewrite = rewrite.split("<|")[0]
            rewrite = rewrite.split("#")[0]
            rewrite = " ".join(rewrite.split())
            return query, answer[0], rewrite


class ReadabilityModelWrapper(object):
    """
        ReadabilityModelWrapper

        Infers the readability of a sentence or phrase by taking the mean of an
        ensemble prediction of models trained off of the same base model when
        using the predict method.
    """

    def __init__(
        self, model_paths=None, base_model=ROBERTA_PATH, device=DEVICE, config=LitModel
    ):
        if model_paths is None:
            model_paths = READABILITY_MODELS
        self.model_paths = model_paths
        self.base_model = base_model
        self.device = device
        self.config = config
        self.models = []
        self.tokenizer = None
        self._load()

    def _load(self):
        for model_path in self.model_paths:
            self.models.append(self._load_readability_model(model_path))
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)

    def _load_readability_model(self, model_path):
        model = self.config()
        model.load_state_dict(torch.load(model_path))
        model.to(self.device)
        return model

    def _infer_partial_readability(self, model, encoded_input) -> float:
        input_ids = encoded_input["input_ids"].to(self.device)
        attention_mask = encoded_input["attention_mask"].to(self.device)
        pred = model(input_ids, attention_mask)
        result = pred.flatten().to("cpu")
        return result.numpy()[0]

    def compare(self, input_prompt, output_prompt):
        input_readability = self.predict(input_prompt)
        output_readability = self.predict(output_prompt)
        delta_readability = output_readability - input_readability
        return input_readability, output_readability, delta_readability

    def predict(self, prompt):
        with torch.no_grad():
            encoded_input = self.tokenizer(
                prompt, padding=True, truncation=True, return_tensors="pt"
            )
            readability = []
            for model in self.models:
                readability.append(
                    self._infer_partial_readability(model, encoded_input)
                )
        return sum(readability) / len(readability)


class SimilarityModelWrapper(object):
    """
        SimilarityModelWrapper

        Uses semantic embedding models to determine the similarity between two sentences when using the
        predict method
    """

    def __init__(self, model_id_or_path=SEMANTIC_EMBEDDING, device=DEVICE):
        self.model_id_or_path = model_id_or_path
        self.device = device
        self.model = None
        self.tokenizer = None
        self._load()

    def _load(self):
        self.model = AutoModel.from_pretrained(self.model_id_or_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id_or_path)

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        # Mean Pooling - Take attention mask into account for correct averaging
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def predict(self, input_prompt, output_response) -> float:
        with torch.no_grad():
            encoded_input = self.tokenizer(
                input_prompt, padding=True, truncation=True, return_tensors="pt"
            ).to(self.device)
            encoded_output = self.tokenizer(
                output_response, padding=True, truncation=True, return_tensors="pt"
            ).to(self.device)
            sim_output = self.model(**encoded_input)
            sentence_embeddings_input = self.mean_pooling(
                sim_output, encoded_input["attention_mask"]
            )
            sim_output = self.model(**encoded_output)
            sentence_embeddings_output = self.mean_pooling(
                sim_output, encoded_output["attention_mask"]
            )
            # Normalize embeddings
            sentence_embeddings_input = F.normalize(
                sentence_embeddings_input, p=2, dim=1
            )
            sentence_embeddings_output = F.normalize(
                sentence_embeddings_output, p=2, dim=1
            )
            similarity = torch.cosine_similarity(
                sentence_embeddings_input, sentence_embeddings_output
            ).to("cpu")
            return similarity.numpy()[0]
