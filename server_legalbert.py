import os
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
import torch

app = FastAPI()
MODEL_NAME = "cointegrated/rubert-tiny2"
token = os.getenv("HUGGINGFACE_HUB_TOKEN")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=token)
model = AutoModel.from_pretrained(MODEL_NAME, token=token)

class EmbeddingRequest(BaseModel):
    text: str

class EmbeddingBatchRequest(BaseModel):
    texts: list

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = mean_pooling(outputs, inputs['attention_mask'])
    return embedding.squeeze().numpy().tolist()

def get_embeddings_batch(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = mean_pooling(outputs, inputs['attention_mask'])
    return embeddings.cpu().numpy().tolist()

@app.post("/embed/")
async def embed(req: EmbeddingRequest):
    emb = get_embedding(req.text)
    return {"embedding": emb}

@app.post("/embed_batch/")
async def embed_batch(req: EmbeddingBatchRequest):
    embs = get_embeddings_batch(req.texts)
    return {"embeddings": embs}
