# main.py
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from dotenv import load_dotenv
import uvicorn
from api.routes import auth, predictions, history
from keep_alive import start_keep_alive
from services.database import ensure_indexes
from services.predictor import JewelryPredictor

load_dotenv()

MAX_UPLOAD_SIZE = int(os.getenv("MAX_UPLOAD_SIZE_MB", 10)) * 1024 * 1024

# H-002: rate limiter (keyed by IP)
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_indexes()
    # C-001: load models once, cache in app.state
    app.state.predictor = JewelryPredictor()
    start_keep_alive(app)
    yield
    app.state.predictor = None


app = FastAPI(
    title="Jewelify API",
    description="API for Jewelify application",
    version="1.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# H-006: File upload size limit middleware
@app.middleware("http")
async def limit_upload_size(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_UPLOAD_SIZE:
        return JSONResponse(
            status_code=413,
            content={"detail": f"Upload exceeds maximum allowed size of {MAX_UPLOAD_SIZE // (1024*1024)} MB"},
        )
    return await call_next(request)


app.include_router(auth.router)
app.include_router(predictions.router)
app.include_router(history.router)


@app.get("/")
async def home():
    return {"Message": "Welcome to Jewelify home page"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)
