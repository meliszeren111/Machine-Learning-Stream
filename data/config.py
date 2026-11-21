
import os

def get_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default

# FastAPI
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = get_int_env("API_PORT", 8000)

# Frontend
FRONTEND_HOST = os.getenv("FRONTEND_HOST", "0.0.0.0")
FRONTEND_PORT = get_int_env("PORT", 10000)

# FastAPI URL
API_URL = os.getenv(
    "API_URL",
    f"http://127.0.0.1:{API_PORT}"
).rstrip("/")
