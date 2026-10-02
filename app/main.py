from fastapi import FastAPI
from pathlib import Path
from typing import Optional
from tensorflow.keras.models import load_model
# pyrefly: ignore [missing-import]
from tensorflow.keras.preprocessing.sequence import pad_sequences
# pyrefly: ignore [missing-import]
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from contextlib import asynccontextmanager
import json
import numpy as np

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


def predict(query):
    sequences = AI_TOKENIZER.texts_to_sequences([query]) # function from keras
    maxlen = MODEL_METADATA.get("max_sequence")
    x_input = pad_sequences(sequences, maxlen = maxlen)
    preds_arr = AI_MODEL.predict(x_input) # array of preds
    preds = (preds_arr[0])
    top_idx_val = np.argmax(preds) # return the idx with highest val
    top_pred = {
        "labels":labels_legend_inverted[str(top_idx_val)],
        "confidence": float(preds[top_idx_val])
    } # store the highest probability only
    labeled_preds = [{"labels":labels_legend_inverted[str(i)],"confidence": float(x)} for i, x in enumerate(list(preds))]
    print(f"labeled preds: {labeled_preds}") # labeled pred would have both
    return {"top": top_pred, "predictions": labeled_preds}


@app.get("/") # run the function below when someone sends a GET request
def read_index(q:Optional[str] = None):
    global AI_MODEL, MODEL_METADATA
    query = q or "hello world"
    preds_dict = predict(query)
    return {"query": query, "results": preds_dict}

