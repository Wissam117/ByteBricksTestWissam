# app/main.py

import os
import sys
import traceback
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time

# Import core configuration
try:
    from app.core.config import settings
except ImportError as e:
    print(f"Error importing config: {e}")
    sys.exit(1)

# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Basic health check endpoint (always available)
@app.get("/")
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": settings.PROJECT_NAME}

# Try to import and register API routes
try:
    from app.api.auth import router as auth_router
    app.include_router(auth_router, prefix=f"{settings.API_V1_STR}", tags=["auth"])
    print("Successfully loaded auth routes")
except ImportError as e:
    print(f"Warning: Could not load auth routes: {e}")
    traceback.print_exc()

try:
    from app.api.concierge import router as concierge_router
    app.include_router(concierge_router, prefix=f"{settings.API_V1_STR}/concierge", tags=["concierge"])
    print("Successfully loaded concierge routes")
except ImportError as e:
    print(f"Warning: Could not load concierge routes: {e}")
    traceback.print_exc()

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    print(f"Global exception: {exc}")
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"detail": f"An unexpected error occurred: {str(exc)}"}
    )

# Additional diagnostic endpoint
@app.get("/debug/info")
def debug_info():
    """Debug information endpoint."""
    return {
        "python_version": sys.version,
        "working_directory": os.getcwd(),
        "environment_variables": {
            "LOCAL_MODEL_PATH": "/mnt/g/Wissam/ByteBricksTestWissam/app/models/Llama-3.2-3B-Instruct-IQ4_XS.gguf",
            "OPENAI_API_KEY": "***" if os.getenv("OPENAI_API_KEY") else None,
            "HUGGINGFACE_API_TOKEN": "***" if os.getenv("HUGGINGFACE_API_TOKEN") else None,
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("Starting uvicorn server...")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)