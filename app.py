import io
import os
import json
import pandas as pd
import streamlit as st
from groq import Groq

st.set_page_config(page_title="CET College Eligibility App", page_icon="🎓", layout="wide")


def get_groq_client():
    api_key = os.getenv("CET_Rank")
    if not api_key:
        try:
            api_key = st.secrets["CET_Rank"]
        except Exception:
            api_key = None

    if not api_key:
        return None

    return Groq(api_key=api_key)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(col).strip().lower().replace(" ", "_") for col in df.columns]
    return df


def validate_cutoff_df(df: pd.DataFrame):
    required_columns = {"college_name", "course", "city", "category", "cutoff_rank"}
    missing = required_columns - set(df.columns)
    return list(missing)


def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    elif file_name.endswith(".txt"):
        # try comma-separated first
        uploaded_file.seek(0)
        try:
            return pd.read_csv(uploaded_file)
        except Exception:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, sep="\t")

    elif file_name.endswith(".tsv"):
        return pd.read_csv(uploaded_file, sep="\t")

    elif file_name.endswith(".xlsx") or file_name.endswith(".xls"):
        return pd.read_excel(uploaded_file)

    elif file_name.endswith(".json"):
        data = json.load(uploaded_file)
        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict):
            return pd.DataFrame(data)
        else:
            raise ValueError("Unsupported JSON structure.")

    else:
        raise ValueError("Unsupported file type. Please upload CSV, TXT, TSV, XLSX, XLS, or JSON.")


def get_eligible_colleges(df: pd.DataFrame, student_rank: int, student_category: str) -> pd.DataFrame:
    filtered_df = df.copy()

    filtered_df["category"] = filtered_df["category"].astype(str).str.upper().str.strip()
    filtered_df["cutoff_rank"] = pd.to_numeric(filtered_df["cutoff_rank"], errors="coerce")
    filtered_df = filtered_df.dropna(subset=["cutoff_rank"])

    filtered_df = filtered_df[filtered_df["category"] == student_category.upper()]
    eligible_df = filtered_df[filtered_df["cutoff_rank"] >= student_rank].copy()

    eligible_df = eligible_df.sort_values(by="cutoff_rank", ascending=True)
    eligible_df["student_rank"] = student_rank
    eligible_df["student_category"] = student_category.upper()
    eligible_df["eligibility_status"] = "Eligible to Apply"

    return eligible_df


def build_context_from_df(df: pd.DataFrame, selected_college: str, student_rank: int, student_category: str) -> str:
    context_lines = [
        f"Student CET Rank: {student_rank}",
        f"Student Category: {student_category}",
    ]

    if selected_college:
        selected_rows = df[df["college_name"].astype(str) == selected_college]

        if not selected_rows.empty:
            context_lines.append("Selected College Data:")
            for _, row in selected_rows.iterrows():
                context_lines.append(
                    f"College Name: {row.get('college_name', '')}, "
                    f"Course: {row.get('course', '')}, "
                    f"City: {row.get('city', '')}, "
                    f"Category: {row.get('category', '')}, "
                    f"Cutoff Rank: {row.get('cutoff_rank', '')}"
                )

    return "\n".join(context_lines)


def ask_groq_about_college(
    question: str,
    df: pd.DataFrame,
    selected_college: str,
    student_rank: int,
    student_category: str
) -> str:
    client = get_groq_client()
    if client is None:
        raise ValueError("GROQ_API_KEY is missing. Add it in Streamlit secrets or environment variables.")

    context = build_context_from_df(df, selected_college, student_rank, student_category)

    system_prompt = (
        "You are a CET college admission assistant. "
        "Answer only using the uploaded cutoff data and student details provided to you. "
        "Keep the answer simple, short, and clear. "
        "If the answer is not available in the dataset, clearly say that it is not available."
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


if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Upload the cutoff file, enter CET rank and category, view eligible colleges, and ask questions about a college.",
        }
    ]


st.title("🎓 CET College Eligibility and College Chat App")
st.write(
    "Upload the cutoff file, enter student rank and category, get eligible colleges, "
    "chat about a college, and download the eligible college list as CSV."
)

with st.expander("Expected Data Format"):
    st.write("Your uploaded file should contain these columns:")
    st.code("college_name, course, city, category, cutoff_rank", language="text")

    sample_csv = """college_name,course,city,category,cutoff_rank
BMS College of Engineering,ISE,Bengaluru,GM,5500
BMS College of Engineering,ISE,Bengaluru,OBC,8500
RNS Institute of Technology,AIML,Bengaluru,GM,12000
NCET Bangalore,CSE,Bengaluru,SC,70000
"""
    st.code(sample_csv, language="csv")


with st.sidebar:
    st.header("Student Input")
    uploaded_file = st.file_uploader(
        "Upload Cutoff File",
        type=["csv", "txt", "tsv", "xlsx", "xls", "json"]
    )
    student_rank = st.number_input("Enter CET Rank", min_value=1, step=1)
    student_category = st.selectbox("Select Category", ["GM", "OBC", "SC", "ST"])
    st.info("The final result can be downloaded as CSV.")


cutoff_df = None
eligible_df = pd.DataFrame()
selected_college = None

if uploaded_file is not None:
    try:
        cutoff_df = read_uploaded_file(uploaded_file)
        cutoff_df = normalize_columns(cutoff_df)

        missing_columns = validate_cutoff_df(cutoff_df)
        if missing_columns:
            st.error("Missing required columns: " + ", ".join(missing_columns))
        else:
            eligible_df = get_eligible_colleges(cutoff_df, int(student_rank), student_category)

    except Exception as e:
        st.error(f"Error reading uploaded file: {e}")


col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("Eligible Colleges")

    if uploaded_file is None:
        st.warning("Please upload the cutoff file first.")
    elif cutoff_df is not None and eligible_df.empty:
        st.info("No eligible colleges found for this student rank and category.")
    else:
        display_columns = [
            col for col in
            ["college_name", "course", "city", "category", "cutoff_rank", "eligibility_status"]
            if col in eligible_df.columns
        ]

        st.dataframe(eligible_df[display_columns], use_container_width=True)

        unique_colleges = eligible_df["college_name"].dropna().astype(str).unique().tolist()

        if unique_colleges:
            selected_college = st.selectbox("Select a college to chat about", ["Select"] + unique_colleges)
            if selected_college == "Select":
                selected_college = None

        csv_buffer = io.StringIO()
        eligible_df.to_csv(csv_buffer, index=False)

        st.download_button(
            label="Download Eligible Colleges CSV",
            data=csv_buffer.getvalue(),
            file_name="eligible_colleges.csv",
            mime="text/csv",
        )


with col2:
    st.subheader("College Chat Assistant")

    if selected_college:
        st.success(f"Selected College: {selected_college}")
    else:
        st.info("Select a college from the eligible college list to ask questions.")

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
                raise ValueError("Please upload the cutoff file first.")

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
st.subheader("Files Needed")

st.code(
    """requirements.txt
streamlit
pandas
groq
openpyxl
""",
    language="text"
)

st.code(
    """.gitignore
.streamlit/secrets.toml
__pycache__/
*.pyc
.env
""",
    language="text"
)

st.code(
    """.streamlit/secrets.toml
GROQ_API_KEY = "your_groq_api_key_here"
""",
    language="toml"
)

st.caption("Do not upload your real API key to GitHub.")
