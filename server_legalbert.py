import os
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
import torch

app = FastAPI()
MODEL_NAME = "ai-forever/ruLegalBert-base"
token = os.getenv("HUGGINGFACE_HUB_TOKEN")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=token)
model = AutoModel.from_pretrained(MODEL_NAME, token=token)

class EmbeddingRequest(BaseModel):
    text: str

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    return embedding.tolist()

@app.post("/embed/")
async def embed(req: EmbeddingRequest):
    emb = get_embedding(req.text)
    return {"embedding": emb}
