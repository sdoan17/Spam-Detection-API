from fastapi import FastAPI
from pathlib import Path
from typing import Optional
from tensorflow.keras.models import load_model
# pyrefly: ignore [missing-import]
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from contextlib import asynccontextmanager
import json

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR.parent/"models"
MODEL_PATH = MODEL_DIR / "spam-model.keras"
TOKENIZER_PATH = MODEL_DIR / "spam-classifier-tokenizer.json"
METADATA_PATH = MODEL_DIR / "spam-classifier-metadata.json"

AI_MODEL = None
AI_TOKENIZER = None
MODEL_METADATA = {}
labels_legend_inverted = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global AI_MODEL, AI_TOKENIZER, MODEL_METADATA, labels_legend_inverted
    if MODEL_PATH.exists():
        AI_MODEL = load_model(MODEL_PATH)
    if TOKENIZER_PATH.exists():
        AI_TOKENIZER = tokenizer_from_json(TOKENIZER_PATH.read_text())
    if METADATA_PATH.exists():
        MODEL_METADATA = json.loads(METADATA_PATH.read_text())
        labels_legend_inverted = MODEL_METADATA['labels_legend_inverted']
    yield


app = FastAPI(lifespan=lifespan)

@app.get("/") # run the function below when someone sends a GET request
def read_index(q:Optional[str] = None):
    global AI_MODEL, MODEL_METADATA
    query = q or "hello world"
    # predict(query)
    print(AI_MODEL)
    return {"query": query, **MODEL_METADATA, "legend": labels_legend_inverted}

