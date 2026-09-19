import requests
import streamlit as st

API = "http://127.0.0.1:8000"

st.set_page_config(page_title="Support Ticket AI", page_icon="🎫", layout="wide")
st.title("🎫 Support Ticket AI")

try:
    health = requests.get(f"{API}/health", timeout=5).json()
    st.sidebar.success(f"API online — {health['rows_loaded']} tickets")
    st.sidebar.caption(f"LLM: {health['llm_provider']} / {health['llm_model']}")
    st.sidebar.caption(f"Reference date: {health['reference_date'][:16]}")
except Exception as e:
    st.sidebar.error(f"API unreachable: {e}")
    st.stop()

tab1, tab2, tab3 = st.tabs(["💬 Ask a question", "🚨 Anomalies", "📊 Dataset stats"])

with tab1:
    st.subheader("Ask about the tickets in plain English")
    samples = [
        "How many tickets are currently open?",
        "Which agent resolved the most tickets?",
        "Show me all Critical tickets not resolved within 12 hours.",
        "What is the average customer rating for Technical category tickets?",
    ]
    picked = st.selectbox("Sample questions", ["—"] + samples)
    question = st.text_input("Your question", value="" if picked == "—" else picked)

    if st.button("Ask", type="primary") and question.strip():
        with st.spinner("Thinking..."):
            r = requests.post(f"{API}/query", json={"question": question}, timeout=60)
        if r.status_code == 200:
            data = r.json()
            st.success(data["answer"])
            col1, col2, col3 = st.columns(3)
            col1.metric("Matching rows", data["row_count"])
            col2.metric("Latency", f"{data['latency_ms']} ms")
            col3.metric("LLM used", "Yes" if data["llm_used"] else "No")
            with st.expander("Query plan (what the LLM understood)"):
                st.json(data["intent"])
            if isinstance(data["result"], list) and data["result"]:
                st.dataframe(data["result"])
        else:
            st.error(f"Error: {r.text}")

with tab2:
    st.subheader("Flagged anomalies")
    summary = requests.get(f"{API}/anomalies/summary", timeout=10).json()
    st.metric("Total anomalies", summary["total"])
    cols = st.columns(len(summary["by_type"]))
    for col, (name, count) in zip(cols, summary["by_type"].items()):
        col.metric(name.replace("_", " ").title(), count)

    anomalies = requests.get(f"{API}/anomalies?limit=200", timeout=10).json()
    st.dataframe([
        {"ticket_id": a["ticket_id"], "type": a["type"], "severity": a["severity"], "reason": a["reason"]}
        for a in anomalies
    ])

with tab3:
    stats = requests.get(f"{API}/stats", timeout=10).json()
    c1, c2 = st.columns(2)
    c1.metric("Total tickets", stats["rows"])
    c2.metric("Agents", stats["agents"])
    col1, col2, col3 = st.columns(3)
    col1.bar_chart(stats["status_counts"])
    col1.caption("By status")
    col2.bar_chart(stats["priority_counts"])
    col2.caption("By priority")
    col3.bar_chart(stats["category_counts"])
    col3.caption("By category")