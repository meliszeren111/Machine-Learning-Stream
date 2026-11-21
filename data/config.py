import os

def get_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


# Backend
API_HOST = "0.0.0.0"
API_PORT = get_int_env("PORT", 8000)

# Frontend
FRONTEND_HOST = "0.0.0.0"
FRONTEND_PORT = get_int_env("PORT", 10000)

# Backend URL
API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000"
).rstrip("/")
