uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload & streamlit run frontend/app.py

streamlit run frontend/app.py
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000