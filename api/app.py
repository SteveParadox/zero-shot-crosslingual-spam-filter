from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
import torch
import logging

app = FastAPI()

# Prometheus Instrumentation
Instrumentator().instrument(app).expose(app)

# Rate Limiting Middleware
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, lambda request, exc: HTTPException(status_code=429, detail="Rate limit exceeded."))
app.add_middleware(SlowAPIMiddleware)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model and tokenizer
MODEL_PATH = "model/saved_model"
LABELS = {0: "ham", 1: "spam"}

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Model loaded on {device}")
except Exception as e:
    logger.error("Failed to load model or tokenizer", exc_info=True)
    raise

# Schema
class TextInput(BaseModel):
    text: str

@app.post("/predict")
@limiter.limit("5/minute")
async def predict(request: Request, input_data: TextInput):
    text = input_data.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.softmax(outputs.logits, dim=1).squeeze().tolist()
        predicted_class = int(torch.argmax(outputs.logits, dim=1))

        return {
            "text": text,
            "label": predicted_class,
            "label_name": LABELS[predicted_class],
            "confidence": round(probs[predicted_class], 4),
            "probabilities": {LABELS[i]: round(p, 4) for i, p in enumerate(probs)}
        }

    except Exception as e:
        logger.error("Prediction error", exc_info=True)
        raise HTTPException(status_code=500, detail="Prediction failed.")
