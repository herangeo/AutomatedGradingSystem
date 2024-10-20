import streamlit as st
import pdfplumber
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from safetensors.torch import load_file
import pandas as pd
import plotly.express as px
import os
import google.generativeai as genai
import plotly.graph_objects as go

# Set the GOOGLE_APPLICATION_CREDENTIALS environment variable directly
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "praxis-practice-420710-4f1884d8cdea.json"

# Set the API key directly
api_key = "AIzaSyD57m_U1H8E4f9zqvS33D_TRxPjSc_E7Ro"
genai.configure(api_key=api_key)

# Initialize the generative model
model = genai.GenerativeModel("gemini-pro")
chat = model.start_chat(history=[])

# Load model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
    model_weights = load_file('./my_model_new/model.safetensors')
    model.load_state_dict(model_weights, strict=False)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

labels = {
    0: "Incorrect",
    1: "Partially correct/Incomplete",
    2: "Correct"
}

def grade_answer(model_answer, student_answer):
    inputs = tokenizer(model_answer, student_answer, padding="max_length", truncation=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    preds = torch.nn.functional.softmax(logits, dim=1)
    preds = preds.numpy().ravel().tolist()
    return {l: p for p, l in zip(preds, labels.values())}

def extract_answers_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()

    # Split text based on the pattern "1.", "2.", etc.
    answers = {}
    for line in text.splitlines():
        if line.strip():
            if line[0].isdigit() and line[1] == '.':
                q_num = line.split('.', 1)[0].strip()
                answer = line.split('.', 1)[1].strip()
                answers[q_num] = answer
    return answers

def generate_feedback(desired_answer, student_answer, grade):
    if grade == "Correct":
        return "-"
    else:
        prompt = f"Compare the following student answer with the model answer and explain in less than 50 words as a paragraph why the grade is {grade}:\n\nModel Answer: {desired_answer}\nStudent Answer: {student_answer}"
        response = chat.send_message(prompt, stream=False)
        return response.text.strip()

# Custom CSS for white background
st.markdown(
    """
    <style>
    body {
        background-color:rgb(9, 110, 86);
        color: white;
    }
    .stApp {
        background-color:rgb(9, 110, 86);
        color: white;
    }
    .title-text {
        color: white;
    }
    .st-emotion-cache-1erivf3 {
    display: flex;
    -webkit-box-align: center;
    align-items: center;
    padding: 1rem;
    background-color: rgb(1, 33, 33);
    border-radius: 0.5rem;
    color: rgb(250, 250, 250);
}
.st-emotion-cache-1avcm0n {
    position: fixed;
    top: 0px;
    left: 0px;
    right: 0px;
    height: 2.875rem;
    background:rgb(9, 110, 86);
    outline: none;
    z-index: 999990;
    display: block;
}
    .stButton > button {
        width: 100%;  /* Make button take up full width of its container */
        height: 50px;  /* Set a fixed height for the buttons */
        border: none;  /* Remove the border */
        background-color: rgb(9, 110, 86);  /* Set button background color */
        color: white;  /* Set button text color */
        font-size: 16px;  /* Increase font size */
        font-weight: bold;  /* Make text bold */
        border-radius: 8px;  /* Slightly round the corners */
        margin-bottom: 10px;  /* Add space between buttons */
    }
    .stButton > button:hover {
        background-color: rgb(9, 110, 86);  /* Change background color on hover */
        color: white;  /* Keep text color white on hover */
    }
    .st-emotion-cache-1cypcdb {
    position: relative;
    top: 2px;
    background-color: rgb(9, 100, 86);
    z-index: 999991;
    min-width: 244px;
    max-width: 550px;
    transform: none;
    transition: transform 300ms, min-width 300ms, max-width 300ms;
}
.st-emotion-cache-13k62yr{
    background-color: rgb(9, 110, 86);
}
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar navigation
if st.sidebar.button("Automated Grading"):
    st.session_state.page = "grading"

if st.sidebar.button("OCR"):
    st.session_state.page = "ocr"

# Default to the "Automated Grading" page
if "page" not in st.session_state:
    st.session_state.page = "grading"

# Display the selected page
if st.session_state.page == "grading":
    st.markdown('<h1 class="title-text">Automated Grading System</h1>', unsafe_allow_html=True)
    st.write("Upload the answer key PDF and multiple student answer PDFs.")
    answer_key_file = st.file_uploader("Upload the Answer Key PDF", type="pdf")
    student_files = st.file_uploader("Upload Student Answer PDFs", type="pdf", accept_multiple_files=True)

    if answer_key_file and student_files:
        answer_key = extract_answers_from_pdf(answer_key_file)

        all_results = []
        question_correct_counts = {}
        all_grades = []

        for student_file in student_files:
            student_answers = extract_answers_from_pdf(student_file)
            student_results = []
            for q_num, student_answer in student_answers.items():
                model_answer = answer_key.get(q_num, "")
                if model_answer:
                    grade = grade_answer(model_answer, student_answer)
                    all_grades.extend(grade.values())
                    final_grade = max(grade, key=grade.get)
                    feedback = generate_feedback(model_answer, student_answer, final_grade)
                    all_grades.append(final_grade)
                    student_results.append({
                        "Question": f"Q{q_num}",
                        "Student Answer": student_answer,
                        "Grade": final_grade,
                        "Feedback": feedback
                    })
                    # Count correct answers
                    if final_grade == "Correct":
                        if q_num not in question_correct_counts:
                            question_correct_counts[q_num] = 0
                        question_correct_counts[q_num] += 1
                else:
                    student_results.append({
                        "Question": f"Q{q_num}",
                        "Student Answer": student_answer,
                        "Grade": "No model answer",
                        "Feedback": "No feedback available"
                    })

            # Add results for this student
            if student_results:
                df = pd.DataFrame(student_results)
                st.write(f"### {student_file.name}")
                st.table(df)

                # CSV Export
                csv = df.to_csv(index=False)
                st.download_button(label="Download Table as CSV", data=csv, file_name=f"{student_file.name}_results.csv", mime="text/csv")

        # Donut Chart for Grade Distribution
        grade_distribution = pd.DataFrame({
            "Label": list(labels.values()),
            "Count": [all_grades.count(l) for l in labels.values()]
        })

        fig_donut = px.pie(grade_distribution, names='Label', values='Count', hole=0.3, title="Grade Distribution")
        st.plotly_chart(fig_donut)

        # Bar Chart for Correct Answers per Question
        question_df = pd.DataFrame(list(question_correct_counts.items()), columns=["Question", "Correct Count"])

        fig_bar = px.bar(question_df, x="Question", y="Correct Count", title="Number of Correct Answers per Question")
        st.plotly_chart(fig_bar)

elif st.session_state.page == "ocr":
    st.markdown('<h1 class="title-text">OCR Page</h1>', unsafe_allow_html=True)
    st.write("This page is under construction.")
