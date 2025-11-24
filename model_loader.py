from typing import List

import torch
from transformers import AutoModel, XLMRobertaTokenizer

from config import DEVICE, MODEL_NAME

_device = torch.device("cuda" if DEVICE == "cuda" and torch.cuda.is_available() else "cpu")

# Use XLMRobertaTokenizer instead of AutoTokenizer
LOCAL_MODEL = MODEL_NAME.startswith("./")

_tokenizer = XLMRobertaTokenizer.from_pretrained(
    MODEL_NAME,
    local_files_only=LOCAL_MODEL
)

_model = AutoModel.from_pretrained(
    MODEL_NAME,
    local_files_only=LOCAL_MODEL
).to(_device)
_model.eval()


class EmbeddingModel:
    def __init__(self) -> None:
        self.tokenizer = _tokenizer
        self.model = _model
        self.device = _device

    def embed(self, text: str) -> List[float]:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            normalized = torch.nn.functional.normalize(cls_embedding, p=2, dim=1)

        return normalized.squeeze(0).tolist()


embedding_model = EmbeddingModel()
