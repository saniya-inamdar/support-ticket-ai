# Support Ticket AI

Natural-language querying and anomaly detection over a customer support ticket dataset (500 rows), with both a REST API and a Streamlit UI.

## Architecture
Question -> LLM (Groq, only produces a JSON query plan) -> Pydantic validation -> Pandas execution -> Answer

The LLM never sees the raw ticket data, so numbers can never be hallucinated — it only classifies intent, while pandas always does the actual calculation.

## Setup
1. `pip install fastapi uvicorn streamlit pandas requests pydantic`
2. Add your `GROQ_API_KEY` in `app.py` (free key: console.groq.com/keys)
3. Keep `support_tickets.csv` in the same folder

## Run
Terminal 1: `uvicorn app:app --reload` (REST API, http://localhost:8000/docs)
Terminal 2: `streamlit run ui_app.py` (UI, http://localhost:8501)

## Files
- `app.py` — REST API (FastAPI), core logic lives here
- `ui_app.py` — Streamlit web UI
- `support_ticket_ai.ipynb` — step-by-step Jupyter notebook, same logic with detailed explanations
- `support_tickets.csv` — dataset (500 tickets)

## Model
Groq `openai/gpt-oss-20b` (free tier). The system doesn't crash without a key either — a rule-based fallback parser kicks in automatically.

## Example queries
- "How many tickets are currently open?" -> 111 tickets
- "Which agent resolved the most tickets?" -> AGT-09, 37 tickets
- "Show me all Critical tickets not resolved within 12 hours." -> 3 tickets

## Anomaly detection
140 anomalies detected across 5 rules: long resolution time (IQR-based), stale high-priority tickets, slow first response (P95), low satisfaction, agent outliers.

## Known limitations
- The dataset ends in March 2024, so relative-time queries like "this week" are evaluated against the dataset's last date rather than the real clock
- Only supports single-hop queries, not multi-step comparisons
