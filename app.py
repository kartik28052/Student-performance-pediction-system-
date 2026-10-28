"""Step 3: The web app. Run with:  streamlit run app.py"""
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="AI Student Performance System", page_icon="🎓")
st.title("🎓 AI Based Student Performance System")
st.write("Enter a student's details to predict their performance level.")

model = joblib.load("model.pkl")

tab1, tab2 = st.tabs(["Predict one student", "Dataset overview"])

with tab1:
    name = st.text_input("Student name (optional)")
    col1, col2 = st.columns(2)
    with col1:
        attendance = st.slider("Attendance (%)", 0, 100, 75)
        study_hours = st.slider("Study hours per day", 0.0, 10.0, 3.0, 0.5)
        sleep_hours = st.slider("Sleep hours per night", 3.0, 12.0, 7.0, 0.5)
    with col2:
        prev_marks = st.slider("Previous exam marks (%)", 0, 100, 60)
        assignments = st.slider("Assignment completion (%)", 0, 100, 70)

    if st.button("Predict performance"):
        row = pd.DataFrame(
            [[attendance, study_hours, prev_marks, assignments, sleep_hours]],
            columns=["attendance", "study_hours", "prev_marks", "assignments", "sleep_hours"],
        )
        result = model.predict(row)[0]
        probs = model.predict_proba(row)[0]
        who = name if name else "This student"

        if result == "High":
            st.success(f"{who}: predicted performance is HIGH 🌟")
        elif result == "Medium":
            st.warning(f"{who}: predicted performance is MEDIUM 📘")
        else:
            st.error(f"{who}: predicted performance is LOW ⚠️ needs support")

        st.bar_chart(pd.DataFrame({"Probability": probs}, index=model.classes_))

        st.subheader("Suggestions")
        tips = []
        if attendance < 75:
            tips.append("Improve attendance to at least 75%.")
        if study_hours < 3:
            tips.append("Increase daily study time to 3+ hours.")
        if assignments < 70:
            tips.append("Complete more assignments on time.")
        if sleep_hours < 6:
            tips.append("Get 6-8 hours of sleep for better focus.")
        if not tips:
            tips.append("Great habits! Keep it up.")
        for t in tips:
            st.write("- " + t)

with tab2:
    df = pd.read_csv("students.csv")
    st.write(f"Total students: {len(df)}")
    st.bar_chart(df["performance"].value_counts())
    st.dataframe(df.head(50))