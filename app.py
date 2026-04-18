import streamlit as st
import pandas as pd
import pdfplumber
import io
import os
from groq import Groq

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="CET College AI App", layout="wide")

def get_groq():
    api_key = os.getenv("CET_Rank") or st.secrets.get("CET_Rank")
    return Groq(api_key=api_key)


# -----------------------------
# LOAD CSV
# -----------------------------
def load_csv(file):
    return pd.read_csv(file)


# -----------------------------
# LOAD PDF TABLE
# -----------------------------
def load_pdf(file):
    tables = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            table = page.extract_table()
            if table:
                df = pd.DataFrame(table[1:], columns=table[0])
                tables.append(df)

    if tables:
        return pd.concat(tables, ignore_index=True)
    return pd.DataFrame()


# -----------------------------
# CLEAN DATA
# -----------------------------
def clean(df):
    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]
    return df


# -----------------------------
# FILTER LOGIC
# -----------------------------
def filter_colleges(df, rank, category):
    df["category"] = df["category"].astype(str).str.upper()
    df["cutoff_rank"] = pd.to_numeric(df["cutoff_rank"], errors="coerce")

    result = df[
        (df["category"] == category.upper()) &
        (df["cutoff_rank"] >= rank)
    ]

    return result.sort_values("cutoff_rank")


# -----------------------------
# GROQ CHAT
# -----------------------------
def chat_with_ai(context, question):
    client = get_groq()

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a CET college admission assistant. Answer clearly and briefly."
            },
            {
                "role": "user",
                "content": f"""
Context:
{context}

Question:
{question}
"""
            }
        ],
        temperature=0.3
    )

    return response.choices[0].message.content


# -----------------------------
# UI
# -----------------------------
st.title("🎓 CET College AI Admission System")

uploaded_file = st.file_uploader("Upload CSV or PDF", type=["csv", "pdf"])

rank = st.number_input("Enter CET Rank", min_value=1, step=1)
category = st.selectbox("Select Category", ["GM", "OBC", "SC", "ST"])

df = None

# -----------------------------
# FILE HANDLING
# -----------------------------
if uploaded_file:

    file_type = uploaded_file.name.split(".")[-1]

    if file_type == "csv":
        df = load_csv(uploaded_file)

    elif file_type == "pdf":
        df = load_pdf(uploaded_file)

    if df is not None and not df.empty:

        df = clean(df)

        st.subheader("📄 Dataset")
        st.dataframe(df)

        required = {"college_name", "course", "city", "category", "cutoff_rank"}

        if not required.issubset(df.columns):
            st.error(f"Missing columns: {required - set(df.columns)}")

        else:
            result = filter_colleges(df, rank, category)

            st.subheader("✅ Eligible Colleges")

            if result.empty:
                st.warning("No colleges found")
            else:
                st.dataframe(result)

                # Download CSV
                csv_buffer = io.StringIO()
                result.to_csv(csv_buffer, index=False)

                st.download_button(
                    "⬇ Download CSV",
                    csv_buffer.getvalue(),
                    "eligible_colleges.csv",
                    "text/csv"
                )

                # -----------------------------
                # CHAT SECTION
                # -----------------------------
                st.subheader("🤖 Ask About Colleges (Groq AI)")

                context = result.to_string()

                question = st.text_input("Ask a question about colleges")

                if question:
                    answer = chat_with_ai(context, question)
                    st.success(answer)

else:
    st.info("Upload file to start")


st.caption("Built using Streamlit + Groq AI + Python")
