import os
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
import torch

app = FastAPI()
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
token = os.getenv("HUGGINGFACE_HUB_TOKEN")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=token)
model = AutoModel.from_pretrained(MODEL_NAME, token=token)

class EmbeddingRequest(BaseModel):
    text: str

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        # Mean pooling (рекомендуется для этой модели)
        embeddings = outputs.last_hidden_state
        attention_mask = inputs['attention_mask']
        mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        embedding = sum_embeddings / sum_mask
        return embedding.squeeze().numpy().tolist()

@app.post("/embed/")
async def embed(req: EmbeddingRequest):
    emb = get_embedding(req.text)
    return {"embedding": emb}
