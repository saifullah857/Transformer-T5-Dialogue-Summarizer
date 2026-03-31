from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

# ── Pydantic schema ────────────────────────────────────────────────────────────
class DialogueInput(BaseModel):
    dialogue: str
    max_length: int = 150     # user can override
    min_length: int = 40
    num_beams: int = 4

# ── App initialization ─────────────────────────────────────────────────────────
app = FastAPI(
    title="Text Summarizer",
    description="Text Summarizer using HuggingFace Transformer",
    version="2.0"
)

# ── CORS ───────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Thread pool for blocking inference ────────────────────────────────────────
executor = ThreadPoolExecutor(max_workers=2)

# ── Model & Tokenizer ──────────────────────────────────────────────────────────
drive_path = r"F:\Ai-Ml\Transformer project\saved_summary_model"

print("Loading model and tokenizer...")
model     = T5ForConditionalGeneration.from_pretrained(drive_path)
tokenizer = T5Tokenizer.from_pretrained(drive_path, legacy=False)
print("Model loaded successfully!")

# ── Device setup ───────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")
model = model.to(device)
model.eval()  # disable dropout

# ── Torch optimizations ───────────────────────────────────────────────────────
torch.set_num_threads(4)  # use 4 CPU threads

# Half precision on GPU (speeds up inference 2x on CUDA)
if device.type == "cuda":
    model = model.half()
    print("Using FP16 (half precision) on GPU")

# Compile model for faster inference (PyTorch 2.0+)
try:
    model = torch.compile(model)
    print("Model compiled with torch.compile ✅")
except Exception:
    print("torch.compile not available, skipping")

# ── Warmup (prevents slow first request) ──────────────────────────────────────
print("Warming up model...")
with torch.no_grad():
    dummy = tokenizer(
        "summarize: warming up the model",
        return_tensors="pt",
        max_length=32
    ).to(device)
    model.generate(dummy["input_ids"], max_length=20, num_beams=1)
print("Warmup complete! Server ready ✅")

# ── Text cleaning ──────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s.,!?'-]", "", text)
    return text.strip()

# ── Simple in-memory cache ─────────────────────────────────────────────────────
summary_cache: dict = {}
MAX_CACHE_SIZE = 100

def get_cached_or_none(key: str):
    return summary_cache.get(key)

def set_cache(key: str, value: str):
    if len(summary_cache) >= MAX_CACHE_SIZE:
        # remove oldest entry
        oldest = next(iter(summary_cache))
        del summary_cache[oldest]
    summary_cache[key] = value

# ── Core inference function (runs in thread pool) ─────────────────────────────
def run_inference(dialogue: str, max_length: int, min_length: int, num_beams: int) -> str:
    # Check cache first
    cache_key = f"{dialogue[:100]}_{max_length}_{min_length}_{num_beams}"
    cached = get_cached_or_none(cache_key)
    if cached:
        return cached

    input_text = "summarize: " + dialogue
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding="longest"
    ).to(device)

    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            min_length=min_length,
            length_penalty=1.5,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=3,
            repetition_penalty=2.0,   # penalizes repeated phrases
        )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # Capitalize first letter
    if summary:
        summary = summary[0].upper() + summary[1:]

    set_cache(cache_key, summary)
    return summary

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def home():
    return FileResponse("templates/index.html")


@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):
    start = time.time()

    dialogue = clean_text(dialogue_input.dialogue)
    if not dialogue:
        raise HTTPException(status_code=400, detail="No text provided.")

    if len(dialogue) < 20:
        raise HTTPException(status_code=400, detail="Text too short to summarize.")

    try:
        # Run blocking inference in thread pool so FastAPI stays responsive
        loop = asyncio.get_event_loop()
        summary = await loop.run_in_executor(
            executor,
            run_inference,
            dialogue,
            dialogue_input.max_length,
            dialogue_input.min_length,
            dialogue_input.num_beams,
        )

        elapsed = round(time.time() - start, 2)
        return {
            "summary": summary,
            "time_taken": f"{elapsed}s",
            "word_count_in": len(dialogue.split()),
            "word_count_out": len(summary.split()),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


# ── Health check ───────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": str(device),
        "model": drive_path,
        "cache_size": len(summary_cache),
        "precision": "fp16" if device.type == "cuda" else "fp32",
    }


# ── Cache stats ────────────────────────────────────────────────────────────────
@app.get("/cache/clear")
async def clear_cache():
    summary_cache.clear()
    return {"status": "cache cleared"}