import streamlit as st
import joblib
import matplotlib.pyplot as plt
import numpy as np

# Load trained model
model = joblib.load("smartstudy_model.pkl")

st.title("SmartStudy AI")
st.subheader("AI-Based Personalized Study Recommendation System")

# Inputs
math = st.slider("Math Score",0,100,50)
physics = st.slider("Physics Score",0,100,50)
programming = st.slider("Programming Score",0,100,50)
hours = st.slider("Daily Study Hours",0,6,2)

# Weak subject detection
def weak_subject(m,p,pr):

    scores = {
        "Math":m,
        "Physics":p,
        "Programming":pr
    }

    weak = min(scores,key=scores.get)

    return weak


# Study recommendation
def recommendation(sub):

    if sub == "Math":
        return "Solve 20 math problems daily"

    elif sub == "Physics":
        return "Revise formulas and practice numerical problems"

    elif sub == "Programming":
        return "Practice coding problems and build mini projects"


if st.button("Analyze Performance"):

    weak = weak_subject(math,physics,programming)

    prediction = model.predict([[math,physics,programming,hours]])

    st.success(f"Predicted Quiz Score: {int(prediction[0])}")

    st.warning(f"Weak Subject: {weak}")

    rec = recommendation(weak)

    st.info(f"Recommendation: {rec}")

    # Graph
    subjects = ["Math","Physics","Programming"]
    scores = [math,physics,programming]

    fig, ax = plt.subplots()

    ax.bar(subjects,scores)

    ax.set_ylabel("Scores")

    ax.set_title("Subject Performance")

    st.pyplot(fig)