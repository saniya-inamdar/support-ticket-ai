import json
import re
import time
import logging
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import requests
from pydantic import BaseModel, Field, field_validator, ValidationError
from fastapi import FastAPI, HTTPException, Query

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("support_ticket_ai")

# ====== EDIT THIS ======
GROQ_API_KEY = ""
# ========================

LLM_PROVIDER = "groq"
GROQ_MODEL = "openai/gpt-oss-20b"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "llama3.1:8b"
DATA_PATH = Path("support_tickets.csv")
LLM_TIMEOUT = 30
REFERENCE_DATE_MODE = "auto"

NUMERIC_COLUMNS = {"response_time_hrs", "resolution_time_hrs", "customer_rating", "age_hrs"}
CATEGORICAL_COLUMNS = {"category", "priority", "status", "agent_id"}
TEXT_COLUMNS = {"ticket_id", "issue_summary"}
DATE_COLUMNS = {"created_at"}
ALLOWED_COLUMNS = NUMERIC_COLUMNS | CATEGORICAL_COLUMNS | TEXT_COLUMNS | DATE_COLUMNS

CATEGORY_VALUES = ["Billing", "Technical", "General"]
PRIORITY_VALUES = ["Low", "Medium", "High", "Critical"]
STATUS_VALUES = ["Open", "Resolved", "Escalated"]

STALE_TICKET_HOURS = 24
LOW_RATING_THRESHOLD = 2
IQR_MULTIPLIER = 1.5
SLOW_RESPONSE_PERCENTILE = 0.95
AGENT_SIGMA_MULTIPLIER = 2.0

DISPLAY_COLUMNS = ["ticket_id", "created_at", "category", "priority", "status",
                    "response_time_hrs", "resolution_time_hrs", "agent_id",
                    "customer_rating", "issue_summary"]


# ---------- Data loading ----------
def resolve_reference_date(frame):
    if REFERENCE_DATE_MODE == "now":
        return pd.Timestamp.now()
    return frame["created_at"].max()


def load_and_clean(path):
    frame = pd.read_csv(path)
    frame["created_at"] = pd.to_datetime(frame["created_at"], errors="coerce")
    for col in ("response_time_hrs", "resolution_time_hrs", "customer_rating"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    for col in ("category", "priority", "status", "agent_id"):
        frame[col] = frame[col].astype(str).str.strip()
    reference_date = resolve_reference_date(frame)
    frame["age_hrs"] = ((reference_date - frame["created_at"]).dt.total_seconds() / 3600).round(2)
    frame["is_resolved"] = frame["status"].eq("Resolved")
    return frame, reference_date


df, REFERENCE_DATE = load_and_clean(DATA_PATH)


# ---------- Anomaly detection ----------
def detect_anomalies(frame):
    out = []

    def num(v):
        return None if pd.isna(v) else round(float(v), 2)

    def details(row):
        return {
            "created_at": row["created_at"].strftime("%Y-%m-%d %H:%M"),
            "category": row["category"], "priority": row["priority"],
            "status": row["status"], "agent_id": row["agent_id"],
            "response_time_hrs": num(row["response_time_hrs"]),
            "resolution_time_hrs": num(row["resolution_time_hrs"]),
            "customer_rating": num(row["customer_rating"]),
            "issue_summary": row["issue_summary"],
        }

    resolved = frame["resolution_time_hrs"].dropna()
    if not resolved.empty:
        q1, q3 = resolved.quantile(0.25), resolved.quantile(0.75)
        fence = q3 + IQR_MULTIPLIER * (q3 - q1)
        for _, row in frame[frame["resolution_time_hrs"] > fence].iterrows():
            hrs = row["resolution_time_hrs"]
            out.append({"ticket_id": row["ticket_id"], "type": "long_resolution",
                        "severity": "high" if hrs > fence * 1.5 else "medium",
                        "reason": f"Resolved in {hrs:.1f}h, beyond IQR fence of {fence:.1f}h.",
                        "details": details(row)})

    stale = frame[(frame["status"] != "Resolved") & frame["priority"].isin(["High", "Critical"])
                  & (frame["age_hrs"] > STALE_TICKET_HOURS)]
    for _, row in stale.iterrows():
        out.append({"ticket_id": row["ticket_id"], "type": "stale_high_priority",
                    "severity": "high" if row["priority"] == "Critical" else "medium",
                    "reason": f"{row['priority']} still '{row['status']}' after {row['age_hrs']:.0f}h.",
                    "details": details(row)})

    cutoff = frame["response_time_hrs"].quantile(SLOW_RESPONSE_PERCENTILE)
    for _, row in frame[frame["response_time_hrs"] > cutoff].iterrows():
        out.append({"ticket_id": row["ticket_id"], "type": "slow_first_response",
                    "severity": "medium",
                    "reason": f"First response {row['response_time_hrs']:.1f}h > P95 ({cutoff:.1f}h).",
                    "details": details(row)})

    median_res = frame["resolution_time_hrs"].median()
    unhappy = frame[(frame["customer_rating"] <= LOW_RATING_THRESHOLD)
                    & (frame["resolution_time_hrs"] > median_res)]
    for _, row in unhappy.iterrows():
        out.append({"ticket_id": row["ticket_id"], "type": "low_satisfaction",
                    "severity": "medium" if row["customer_rating"] == 2 else "high",
                    "reason": f"Rating {int(row['customer_rating'])}/5 after {row['resolution_time_hrs']:.1f}h.",
                    "details": details(row)})

    agent_means = frame.groupby("agent_id")["resolution_time_hrs"].mean().dropna()
    if len(agent_means) > 1:
        limit = agent_means.mean() + AGENT_SIGMA_MULTIPLIER * agent_means.std()
        for agent, value in agent_means[agent_means > limit].items():
            out.append({"ticket_id": f"AGENT::{agent}", "type": "agent_outlier",
                        "severity": "medium",
                        "reason": f"{agent} averages {value:.1f}h, above {AGENT_SIGMA_MULTIPLIER}sigma limit {limit:.1f}h.",
                        "details": {"agent_id": agent, "avg_resolution_time_hrs": round(float(value), 2)}})

    order = {"high": 0, "medium": 1, "low": 2}
    out.sort(key=lambda a: (order[a["severity"]], a["ticket_id"]))
    return out


# ---------- LLM client ----------
JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


class LLMUnavailable(RuntimeError):
    pass


class LLMClient:
    def __init__(self, provider=LLM_PROVIDER):
        self.provider = provider
        self.model = GROQ_MODEL if provider == "groq" else OLLAMA_MODEL

    def chat(self, messages, temperature=0.0):
        last_err = None
        for attempt in (1, 2):
            try:
                if self.provider == "groq":
                    return self._groq(messages, temperature)
                return self._ollama(messages, temperature)
            except Exception as exc:
                last_err = exc
                logger.warning("LLM attempt %s failed: %s", attempt, exc)
                time.sleep(0.5 * attempt)
        raise LLMUnavailable(str(last_err))

    def chat_json(self, messages):
        return self._parse_json(self.chat(messages))

    def ping(self):
        try:
            if self.provider == "groq":
                if not GROQ_API_KEY:
                    return False
                self.chat([{"role": "user", "content": "ping"}])
            else:
                requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3).raise_for_status()
            return True
        except Exception:
            return False

    def _groq(self, messages, temperature):
        if not GROQ_API_KEY:
            raise LLMUnavailable("GROQ_API_KEY is not set")
        r = requests.post(GROQ_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={"model": self.model, "messages": messages, "temperature": temperature,
                  "max_tokens": 1024, "reasoning_effort": "low"},
            timeout=LLM_TIMEOUT)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()

    def _ollama(self, messages, temperature):
        r = requests.post(f"{OLLAMA_HOST}/api/chat",
            json={"model": self.model, "messages": messages, "stream": False,
                  "options": {"temperature": temperature}}, timeout=LLM_TIMEOUT)
        r.raise_for_status()
        return r.json()["message"]["content"].strip()

    @staticmethod
    def _parse_json(raw):
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned, flags=re.MULTILINE).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = JSON_BLOCK.search(cleaned)
            if not match:
                raise LLMUnavailable(f"No JSON found in: {raw[:200]}")
            return json.loads(match.group(0))


llm = LLMClient()


# ---------- Query schema ----------
FilterOp = Literal["eq", "neq", "gt", "gte", "lt", "lte", "in", "not_in", "contains", "is_null", "not_null"]
Operation = Literal["count", "average", "sum", "max", "min", "list", "groupby_rank", "distribution"]


class Filter(BaseModel):
    column: str
    op: FilterOp = "eq"
    value: Any = None

    @field_validator("column")
    @classmethod
    def known_column(cls, v):
        if v not in ALLOWED_COLUMNS:
            raise ValueError(f"Unknown column '{v}'. Allowed: {sorted(ALLOWED_COLUMNS)}")
        return v


class QueryIntent(BaseModel):
    operation: Operation
    target_column: str | None = None
    group_by: str | None = None
    filters: list[Filter] = Field(default_factory=list)
    sort: Literal["asc", "desc"] | None = None
    limit: int = Field(default=10, ge=1, le=200)

    @field_validator("target_column", "group_by")
    @classmethod
    def known_optional_column(cls, v):
        if v in (None, "", "null"):
            return None
        if v not in ALLOWED_COLUMNS:
            raise ValueError(f"Unknown column '{v}'")
        return v


# ---------- Prompts ----------
INTENT_SYSTEM_PROMPT = f'''You translate questions about a customer-support \
ticket dataset into a strict JSON query plan. You never answer the question \
yourself and you never invent numbers — a separate pandas engine executes your \
plan against the real data.

COLUMNS
- ticket_id            string, unique
- created_at           datetime
- category             one of {CATEGORY_VALUES}
- priority             one of {PRIORITY_VALUES}
- status               one of {STATUS_VALUES}
- response_time_hrs    float, hours to first response
- resolution_time_hrs  float, null when the ticket is not resolved
- agent_id             string, AGT-01 .. AGT-12
- customer_rating      int 1-5, null when the ticket is not resolved
- issue_summary        free text
- age_hrs              float, hours between created_at and the reference date

OUTPUT SCHEMA (return this object and nothing else)
{{
  "operation": "count|average|sum|max|min|list|groupby_rank|distribution",
  "target_column": "<column or null>",
  "group_by": "<column or null>",
  "filters": [{{"column": "<column>", "op": "eq|neq|gt|gte|lt|lte|in|not_in|contains|is_null|not_null", "value": <any>}}],
  "sort": "asc|desc|null",
  "limit": <int 1-200>
}}

RULES
1. Output raw JSON only. No markdown fences, no commentary.
2. Use only the columns listed above. Never invent a column name.
3. "unresolved", "still open", "not closed" -> filter status neq "Resolved".
4. "this week"/"last 24 hours"/"older than X hours" -> filter on age_hrs.
5. Ranking questions ("which agent has the lowest/highest ...") ->
   operation "groupby_rank", group_by the entity, target_column the metric,
   sort "asc" for lowest and "desc" for highest, limit 1 unless a number is given.
6. "How many" -> count. "Average/mean" -> average. "Show me/list" -> list.
7. Categorical values are capitalised exactly as listed above.
'''

FEW_SHOT_EXAMPLES = [
    ("How many critical tickets are unresolved?",
     '{"operation":"count","target_column":null,"group_by":null,'
     '"filters":[{"column":"priority","op":"eq","value":"Critical"},'
     '{"column":"status","op":"neq","value":"Resolved"}],"sort":null,"limit":10}'),
    ("Which agent has the lowest average customer rating?",
     '{"operation":"groupby_rank","target_column":"customer_rating",'
     '"group_by":"agent_id","filters":[],"sort":"asc","limit":1}'),
    ("What is the average customer rating for Technical category tickets?",
     '{"operation":"average","target_column":"customer_rating","group_by":null,'
     '"filters":[{"column":"category","op":"eq","value":"Technical"}],"sort":null,"limit":10}'),
    ("Show me all Critical tickets not resolved within 12 hours.",
     '{"operation":"list","target_column":null,"group_by":null,'
     '"filters":[{"column":"priority","op":"eq","value":"Critical"},'
     '{"column":"resolution_time_hrs","op":"gt","value":12}],"sort":"desc","limit":50}'),
    ("How many tickets are currently open?",
     '{"operation":"count","target_column":null,"group_by":null,'
     '"filters":[{"column":"status","op":"eq","value":"Open"}],"sort":null,"limit":10}'),
    ("Which agent resolved the most tickets?",
     '{"operation":"groupby_rank","target_column":"ticket_id","group_by":"agent_id",'
     '"filters":[{"column":"status","op":"eq","value":"Resolved"}],"sort":"desc","limit":1}'),
]


def build_intent_messages(question):
    messages = [{"role": "system", "content": INTENT_SYSTEM_PROMPT}]
    for q, a in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": question})
    return messages


# ---------- Query engine ----------
class QueryError(ValueError):
    pass


def apply_filter(frame, f):
    col, op, val = f.column, f.op, f.value
    series = frame[col]

    if op == "is_null":
        return frame[series.isna()]
    if op == "not_null":
        return frame[series.notna()]
    if op in {"in", "not_in"}:
        values = val if isinstance(val, list) else [val]
        mask = series.isin(values)
        return frame[~mask] if op == "not_in" else frame[mask]
    if op == "contains":
        return frame[series.astype(str).str.contains(str(val), case=False, na=False)]
    if op in {"gt", "gte", "lt", "lte"}:
        num_val = pd.to_numeric(val, errors="coerce")
        if pd.isna(num_val):
            raise QueryError(f"'{val}' is not a number for '{op}' on {col}")
        comparisons = {"gt": series > num_val, "gte": series >= num_val,
                       "lt": series < num_val, "lte": series <= num_val}
        return frame[comparisons[op].fillna(False)]

    if pd.api.types.is_numeric_dtype(series):
        mask = series == pd.to_numeric(val, errors="coerce")
    else:
        mask = series.astype(str).str.lower() == str(val).lower()
    return frame[~mask] if op == "neq" else frame[mask]


def apply_filters(frame, filters):
    for f in filters:
        frame = apply_filter(frame, f)
    return frame


AGG_BY_OP = {"average": "mean", "sum": "sum", "max": "max", "min": "min"}


def execute(intent, frame=None):
    frame = apply_filters((frame if frame is not None else df).copy(), intent.filters)
    matched = len(frame)
    op = intent.operation

    if op == "count":
        if intent.group_by:
            counts = frame.groupby(intent.group_by).size().sort_values(ascending=(intent.sort == "asc"))
            return {"value": counts.head(intent.limit).to_dict(), "row_count": matched}
        return {"value": matched, "row_count": matched}

    if op == "distribution":
        group_col = intent.group_by or "status"
        counts = frame.groupby(group_col).size().sort_values(ascending=(intent.sort == "asc"))
        return {"value": counts.head(intent.limit).to_dict(), "row_count": matched}

    if op in AGG_BY_OP:
        if not intent.target_column:
            raise QueryError(f"'{op}' needs a target_column")
        col, agg = intent.target_column, AGG_BY_OP[op]
        if intent.group_by:
            grouped = frame.groupby(intent.group_by)[col].agg(agg).dropna().sort_values(ascending=(intent.sort == "asc"))
            return {"value": {k: round(float(v), 2) for k, v in grouped.head(intent.limit).items()}, "row_count": matched}
        value = getattr(frame[col], agg)()
        return {"value": None if pd.isna(value) else round(float(value), 2), "row_count": matched}

    if op == "groupby_rank":
        if not intent.group_by:
            raise QueryError("groupby_rank needs a group_by column")
        col = intent.target_column or "ticket_id"
        if col in {"ticket_id", "issue_summary"}:
            series, metric = frame.groupby(intent.group_by).size(), "ticket_count"
        else:
            series, metric = frame.groupby(intent.group_by)[col].mean().dropna(), f"avg_{col}"
        series = series.sort_values(ascending=(intent.sort == "asc"))
        cast = int if metric == "ticket_count" else (lambda x: round(float(x), 2))
        ranked = [{intent.group_by: k, metric: cast(v)} for k, v in series.head(intent.limit).items()]
        return {"value": ranked, "row_count": matched, "metric": metric}

    if op == "list":
        subset = frame
        sort_col = intent.target_column or "created_at"
        if intent.sort:
            subset = subset.sort_values(sort_col, ascending=(intent.sort == "asc"))
        records = subset.head(intent.limit)[DISPLAY_COLUMNS].copy()
        records["created_at"] = records["created_at"].dt.strftime("%Y-%m-%d %H:%M")
        return {"value": records.where(pd.notna(records), None).to_dict("records"), "row_count": matched}

    raise QueryError(f"Unsupported operation: {op}")


def describe(intent, result):
    value, rows, op = result["value"], result["row_count"], intent.operation
    if op == "count" and not intent.group_by:
        return f"{value} ticket(s) match that criteria."
    if value in (None, {}, []):
        return "No tickets matched that criteria."
    if op in AGG_BY_OP and not intent.group_by:
        return f"The {op} {intent.target_column} is {value} (across {rows} matching ticket(s))."
    if op == "groupby_rank":
        metric = result.get("metric", "value")
        top = value[0]
        return f"{top[intent.group_by]} with {metric} = {top[metric]} (out of {rows} matching ticket(s))."
    if op == "list":
        ids = ", ".join(r["ticket_id"] for r in value[:5])
        return f"{rows} ticket(s) matched. First few: {ids}."
    if isinstance(value, dict):
        return ", ".join(f"{k}: {v}" for k, v in value.items()) + "."
    return str(value)


def rule_based_intent(question):
    q = question.lower()
    filters = []
    for value in ("critical", "high", "medium", "low"):
        if re.search(rf"\b{value}\b", q):
            filters.append({"column": "priority", "op": "eq", "value": value.title()})
            break
    for value in ("billing", "technical", "general"):
        if value in q:
            filters.append({"column": "category", "op": "eq", "value": value.title()})
            break
    if "unresolved" in q or "not resolved" in q:
        filters.append({"column": "status", "op": "neq", "value": "Resolved"})
    elif "open" in q:
        filters.append({"column": "status", "op": "eq", "value": "Open"})
    elif "escalated" in q:
        filters.append({"column": "status", "op": "eq", "value": "Escalated"})
    elif "resolved" in q:
        filters.append({"column": "status", "op": "eq", "value": "Resolved"})

    if "agent" in q and any(w in q for w in ("which", "who", "lowest", "highest", "most")):
        lowest = any(w in q for w in ("lowest", "worst", "least", "fewest"))
        target = "customer_rating" if "rating" in q else "ticket_id"
        return QueryIntent(operation="groupby_rank", target_column=target, group_by="agent_id",
                            filters=filters, sort="asc" if lowest else "desc", limit=1)
    if "average" in q or "avg" in q or "mean" in q:
        target = "resolution_time_hrs" if "resolution" in q else ("response_time_hrs" if "response" in q else "customer_rating")
        return QueryIntent(operation="average", target_column=target, filters=filters)
    if q.startswith(("show", "list", "give me")):
        return QueryIntent(operation="list", filters=filters, limit=20)
    return QueryIntent(operation="count", filters=filters)


def answer_question(question, explain=True):
    started = time.perf_counter()
    llm_used, fallback_reason = True, None

    try:
        raw_intent = llm.chat_json(build_intent_messages(question))
        intent = QueryIntent.model_validate(raw_intent)
    except LLMUnavailable as exc:
        llm_used, fallback_reason = False, f"LLM unavailable ({exc})"
        intent = rule_based_intent(question)
    except ValidationError as exc:
        llm_used, fallback_reason = False, f"LLM returned invalid plan ({exc})"
        intent = rule_based_intent(question)

    result = execute(intent)
    answer = describe(intent, result)

    if explain and llm_used:
        try:
            polished = llm.chat([
                {"role": "system", "content": "Turn the computed result into one short factual sentence. Use only the numbers given."},
                {"role": "user", "content": f"QUESTION: {question}\nPLAN: {intent.model_dump()}\nRESULT: {answer}\n\nWrite the answer sentence."},
            ])
            if polished:
                answer = polished.strip()
        except LLMUnavailable:
            pass

    return {"question": question, "answer": answer, "intent": intent.model_dump(),
            "result": result["value"], "row_count": result["row_count"],
            "llm_used": llm_used, "fallback_reason": fallback_reason,
            "latency_ms": int((time.perf_counter() - started) * 1000)}


# ---------- FastAPI app ----------
app = FastAPI(title="Support Ticket AI")


@app.get("/health")
def health():
    return {"status": "ok", "rows_loaded": len(df), "llm_provider": llm.provider,
            "llm_model": llm.model, "llm_reachable": llm.ping(),
            "reference_date": str(REFERENCE_DATE)}


@app.post("/query")
def query_endpoint(payload: dict):
    question = payload.get("question", "")
    explain = payload.get("explain", True)
    if not question:
        raise HTTPException(status_code=400, detail="'question' is required")
    try:
        return answer_question(question, explain)
    except QueryError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/anomalies")
def get_anomalies(type: str | None = None, severity: str | None = None, limit: int = Query(default=50, le=500)):
    found = detect_anomalies(df)
    if type:
        found = [a for a in found if a["type"] == type]
    if severity:
        found = [a for a in found if a["severity"] == severity]
    return found[:limit]


@app.get("/anomalies/summary")
def anomalies_summary():
    found = detect_anomalies(df)
    by_type, by_sev = {}, {}
    for a in found:
        by_type[a["type"]] = by_type.get(a["type"], 0) + 1
        by_sev[a["severity"]] = by_sev.get(a["severity"], 0) + 1
    return {"total": len(found), "by_type": by_type, "by_severity": by_sev}


@app.get("/stats")
def stats():
    return {
        "rows": len(df),
        "status_counts": df["status"].value_counts().to_dict(),
        "priority_counts": df["priority"].value_counts().to_dict(),
        "category_counts": df["category"].value_counts().to_dict(),
        "agents": int(df["agent_id"].nunique()),
    }