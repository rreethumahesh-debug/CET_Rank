import io
import os
import pandas as pd
import streamlit as st
from groq import Groq

st.set_page_config(page_title="CET College Eligibility App", page_icon="🎓", layout="wide")

# =============================
# Helpers
# ======================([console.groq.com](https://console.groq.com/docs/quickstart?utm_source=chatgpt.com))"GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", None)
    if not api_key:
        return None
    return Groq(api_key=CET_Rank)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(col).strip().lower().replace(" ", "_") for col in df.columns]
    return df


def validate_cutoff_df(df: pd.DataFrame):
    required = {"college_name", "course", "city", "category", "cutoff_rank"}
    missing = required - set(df.columns)
    return sorted(list(missing))


def get_eligible_colleges(df: pd.DataFrame, student_rank: int, student_category: str) -> pd.DataFrame:
    filtered = df[df["category"].astype(str).str.upper() == student_category.upper()].copy()
    filtered["cutoff_rank"] = pd.to_numeric(filtered["cutoff_rank"], errors="coerce")
    filtered = filtered.dropna(subset=["cutoff_rank"])

    eligible = filtered[filtered["cutoff_rank"] >= student_rank].copy()
    eligible = eligible.sort_values(by="cutoff_rank", ascending=True)
    eligible["student_rank"] = student_rank
    eligible["student_category"] = student_category.upper()
    eligible["eligibility_status"] = "Eligible to Apply"
    return eligible


def build_context_from_df(df: pd.DataFrame, selected_college: str | None, student_rank: int, student_category: str) -> str:
    lines = [
        f"Student rank: {student_rank}",
        f"Student category: {student_category}",
    ]

    if selected_college:
        college_rows = df[df["college_name"].astype(str) == selected_college]
        if not college_rows.empty:
            lines.append("Selected college data:")
            for _, row in college_rows.head(10).iterrows():
                lines.append(
                    f"College: {row.get('college_name', '')} | Course: {row.get('course', '')} | "
                    f"City: {row.get('city', '')} | Category: {row.get('category', '')} | "
                    f"Cutoff Rank: {row.get('cutoff_rank', '')}"
                )

    return "\n".join(lines)


def ask_groq_about_college(question: str, df: pd.DataFrame, selected_college: str | None, student_rank: int, student_category: str) -> str:
    client = get_groq_client()
    if client is None:
        raise ValueError("GROQ_API_KEY is missing. Add it in Streamlit secrets or environment variables.")

    context = build_context_from_df(df, selected_college, student_rank, student_category)

    system_prompt = (
        "You are a CET college admission assistant. "
        "Answer only from the uploaded cutoff data and the provided student details. "
        "Keep answers simple and clear. "
        "If some information is not present in the uploaded data, clearly say it is not available in the dataset."
    )

    user_prompt = f"""
Uploaded dataset context:
{context}

Student question:
{question}
"""

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=500,
    )

    return completion.choices[0].message.content


# =============================
# Session state
# =============================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Upload the cutoff CSV, enter CET rank and category, view eligible colleges, and ask questions about a college.",
        }
    ]


# =============================
# UI Header
# =============================
st.title("🎓 CET College Eligibility and Chat App")
st.write(
    "Upload a cutoff list CSV, enter student rank and category, get eligible colleges, chat about a college, and download the final eligible list as CSV."
)

with st.expander("Expected CSV format"):
    st.write("Your cutoff CSV should contain these columns:")
    st.code("college_name, course, city, category, cutoff_rank", language="text")
    sample_csv = """college_name,course,city,category,cutoff_rank
BMS College of Engineering,ISE,Bengaluru,GM,5500
BMS College of Engineering,ISE,Bengaluru,OBC,8500
RNS Institute of Technology,AIML,Bengaluru,GM,12000
NCET Bangalore,CSE,Bengaluru,SC,70000
"""
    st.code(sample_csv, language="csv")

# =============================
# Sidebar Inputs
# =============================
with st.sidebar:
    st.header("Student Details")
    uploaded_file = st.file_uploader("Upload cutoff CSV", type=["csv"])
    student_rank = st.number_input("Enter CET Rank", min_value=1, step=1)
    student_category = st.selectbox("Select Category", ["GM", "OBC", "SC", "ST"])
    st.info("Final output will be a downloadable CSV of all eligible colleges.")

cutoff_df = None
eligible_df = pd.DataFrame()
selected_college = None

if uploaded_file is not None:
    try:
        cutoff_df = pd.read_csv(uploaded_file)
        cutoff_df = normalize_columns(cutoff_df)

        missing_columns = validate_cutoff_df(cutoff_df)
        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
        else:
            eligible_df = get_eligible_colleges(cutoff_df, int(student_rank), student_category)
    except Exception as e:
        st.error(f"Could not read the CSV file: {e}")

col1, col2 = st.columns([1.2, 1])

# =============================
# Eligible colleges section
# =============================
with col1:
    st.subheader("Eligible Colleges")

    if uploaded_file is None:
        st.warning("Please upload the cutoff CSV first.")
    elif cutoff_df is not None and eligible_df.empty:
        st.info("No eligible colleges found for this rank and category.")
    elif not eligible_df.empty:
        display_cols = [col for col in ["college_name", "course", "city", "category", "cutoff_rank", "eligibility_status"] if col in eligible_df.columns]
        st.dataframe(eligible_df[display_cols], use_container_width=True)

        unique_colleges = eligible_df["college_name"].dropna().astype(str).unique().tolist()
        if unique_colleges:
            selected_college = st.selectbox("Select a college to chat about", [""] + unique_colleges)
            if selected_college == "":
                selected_college = None

        csv_buffer = io.StringIO()
        eligible_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="Download Eligible Colleges CSV",
            data=csv_buffer.getvalue(),
            file_name="eligible_colleges.csv",
            mime="text/csv",
        )

# =============================
# Chat section
# =============================
with col2:
    st.subheader("College Chat Assistant")

    if selected_college:
        st.success(f"Selected College: {selected_college}")
    else:
        st.info("Select a college from the eligible list to ask focused questions.")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_question = st.chat_input("Ask about the selected college")

    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        try:
            if cutoff_df is None:
                raise ValueError("Please upload the cutoff CSV first.")

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer = ask_groq_about_college(
                        question=user_question,
                        df=cutoff_df,
                        selected_college=selected_college,
                        student_rank=int(student_rank),
                        student_category=student_category,
                    )
                    st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            error_message = f"Error: {str(e)}"
            with st.chat_message("assistant"):
                st.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})

st.divider()
st.subheader("GitHub Project Structure")
st.code(
    """
cet-college-eligibility-app/
│-- app.py
│-- requirements.txt
│-- README.md
│-- .gitignore
└-- .streamlit/
    └-- secrets.toml
""",
    language="bash",
)

st.subheader("requirements.txt")
st.code(
    """
streamlit
pandas
groq
""",
    language="text",
)

st.subheader(".streamlit/secrets.toml")
st.code(
    'GROQ_API_KEY = "CET_Rank"',
    language="toml",
)

st.subheader("How this app works")
st.markdown(
    """
1. Upload the cutoff list CSV.
2. Enter student CET rank and category.
3. App filters colleges where the student's rank is eligible for that category.
4. App displays the eligible colleges.
5. User can chat about a selected college using Groq.
6. User can download the final eligible colleges list as CSV.
"""
)

st.caption("Do not upload your real API key to GitHub. Add it only in Streamlit secrets or environment variables.")
