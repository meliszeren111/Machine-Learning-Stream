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
API_HOST: str = os.getenv("API_HOST", "127.0.0.1")
API_PORT: int = get_int_env("API_PORT", 8000)

# Streamlit
FRONTEND_HOST: str = os.getenv("FRONTEND_HOST", "0.0.0.0")
FRONTEND_PORT: int = get_int_env("PORT", get_int_env("FRONTEND_PORT", 10000))

# (Streamlit -> FastAPI) URL
_connect_host: str = "127.0.0.1" if API_HOST in {"0.0.0.0", "::"} else API_HOST
API_URL: str = os.getenv("API_URL", f"http://{_connect_host}:{API_PORT}").rstrip("/")
