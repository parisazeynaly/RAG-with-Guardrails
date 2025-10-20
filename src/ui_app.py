import streamlit as st
import requests
import json

st.title("RAG with Guardrails — Demo UI")
api_url = st.text_input("API URL", value="http://localhost:8000/ask")
query = st.text_area("Your question", "What is this project about?")
top_k = st.slider("Top K", 1, 10, 4)

if st.button("Ask"):
    payload = {"query": query, "top_k": top_k}
    r = requests.post(api_url, json=payload, timeout=60)
    if r.ok:
        data = r.json()
        st.subheader(f"Decision: {data['decision']}")
        st.write(data["answer"])
        with st.expander("Contexts"):
            st.json(data["contexts"])
        with st.expander("Safety Log"):
            st.json(data["safety_log"])
    else:
        st.error(f"Error: {r.status_code} {r.text}")
