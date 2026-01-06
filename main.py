from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from api.v1.chat import router as chat_router
from api.v1.upload import router as upload_router
from api.v1.health import router as health_router

app = FastAPI(title="FastAPI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(health_router, prefix="/v1")
app.include_router(chat_router, prefix="/v1")
app.include_router(upload_router, prefix="/v1")
app.mount("/static", StaticFiles(directory="static"), name="static")