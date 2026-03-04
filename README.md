# 🎓 SmartStudy AI
### AI-Based Personalized Learning & Study Recommendation System

---

## 📌 Project Overview

**SmartStudy AI** is a beginner-friendly machine learning project that analyzes a student's subject scores and automatically generates a **personalized study plan**. It uses a **Random Forest model** trained on student performance data to predict quiz scores and identify which subjects need the most attention.

The project ships with a beautiful **Streamlit web dashboard** — no coding knowledge required to use it!

---

## 🧩 Problem Statement

Many students study hard but not effectively. They often:

- Spend equal time on strong **and** weak subjects
- Don't know which topics are causing them to fall behind
- Lack a data-driven, personalized roadmap for improvement

**SmartStudy AI** solves this by using real performance data to detect gaps and suggest what to study next.

---

## 🗂️ Project Structure

```
SmartStudyAI/
│
├── student_data.csv     ← 25 rows of sample student performance data
├── model.py             ← Machine learning logic & recommendation engine
├── app.py               ← Streamlit web dashboard (the visual interface)
├── requirements.txt     ← All Python packages needed
└── README.md            ← You are here!
```

---

## 🤖 How the AI Recommendation Works

Here's the step-by-step flow:

```
Student enters scores
        ↓
[model.py] Random Forest model predicts QuizScore
        ↓
[model.py] Weak subjects identified (scores below threshold)
        ↓
[model.py] Personalized tips generated per weak subject
        ↓
[app.py]  Dashboard displays chart + recommendations
```

### Why Random Forest?

A **Random Forest** is an ensemble of many decision trees. Each tree independently predicts the quiz score; the final answer is the average across all trees. This approach:

- Handles small datasets well (we only have 25 records)
- Is robust to outliers and noisy data
- Requires minimal tuning for beginners

### Features Used for Prediction

| Feature          | Description                         |
|-----------------|-------------------------------------|
| MathScore        | Score in Mathematics (0–100)        |
| PhysicsScore     | Score in Physics (0–100)            |
| ProgrammingScore | Score in Programming (0–100)        |
| StudyHours       | Average daily study hours           |

### Target Variable

| Target    | Description                       |
|-----------|-----------------------------------|
| QuizScore | Expected quiz performance (0–100) |

---

## 📊 Dataset Overview (`student_data.csv`)

The dataset contains **25 student records** with the following columns:

```
StudentID, MathScore, PhysicsScore, ProgrammingScore, StudyHours, QuizScore
```

- Scores range from 30 to 96
- Study hours range from 1 to 9 per day
- Represents a realistic spread of beginner to advanced students

---

## 🖥️ Dashboard Features

The Streamlit app provides:

| Feature                    | Description                                           |
|---------------------------|-------------------------------------------------------|
| 📐 Score Sliders           | Enter Math, Physics, Programming scores + Study Hours |
| 📊 Performance Bar Chart   | Visual overview with color-coded weak/strong subjects |
| ⚠️  Weak Subject Detection  | Flags subjects below your chosen threshold            |
| 📚 Study Recommendations   | AI-curated tips for each weak subject                 |
| 🚀 Improvement Potential   | Estimated score gain with extra effort                |
| 🔮 Predicted Quiz Score    | Model's forecast of your next quiz result             |
| 🗃️  Dataset Explorer        | Browse all 25 training records with color highlights  |

---

## ⚙️ Installation Steps

### 1. Make sure Python is installed

You need **Python 3.8 or higher**. Check your version:

```bash
python --version
```

If not installed, download it from: https://python.org/downloads

### 2. Clone or download this project

```bash
# If you have git:
git clone https://github.com/your-username/SmartStudyAI.git

# Or just download and extract the ZIP file
```

### 3. Navigate into the project folder

```bash
cd SmartStudyAI
```

### 4. (Optional but recommended) Create a virtual environment

```bash
# Create a virtual environment
python -m venv venv

# Activate it:
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 5. Install required packages

```bash
pip install -r requirements.txt
```

This installs: `pandas`, `numpy`, `scikit-learn`, `streamlit`, `matplotlib`

---

## ▶️ How to Run the App

```bash
streamlit run app.py
```

Your browser will automatically open to:

```
http://localhost:8501
```

> ✅ **That's it!** Adjust the sliders on the left sidebar and watch your personalized study plan update in real time.

---

## 🧪 Testing the ML Model Independently

To run only the backend model and see recommendations in the terminal:

```bash
python model.py
```

You'll see output like:

```
✅ Loaded dataset with 25 student records.
🎯 Model trained! Mean Absolute Error: 3.12 points

📋 Student Input:
   Math: 55  |  Physics: 48  |  Programming: 70  |  Study Hours: 3

🔮 Predicted Quiz Score: 53.8

⚠️  Weak Subjects Detected: Math, Physics

📚 Personalized Study Recommendations:
  [Math]
    📐 Review algebra fundamentals: equations, inequalities, and functions.
    ...

🚀 Improvement Potential: +18.5 points with consistent effort!
```

---

## 📦 Requirements

```
pandas==2.1.4
numpy==1.26.4
scikit-learn==1.4.0
streamlit==1.31.1
matplotlib==3.8.2
```

---

## 🔧 Customization Ideas

Once you're comfortable with the project, try these improvements:

- **Add more subjects** (Chemistry, Biology, History)
- **Expand the dataset** with real student data from your school
- **Add a login system** so multiple students can track progress over time
- **Export recommendations** as a PDF study schedule
- **Connect to a database** (SQLite or Firebase) for persistent data storage
- **Add a progress tracker** to chart improvement over multiple weeks

---

## 📖 Key Concepts for Beginners

| Concept             | What it means in this project                           |
|--------------------|---------------------------------------------------------|
| Random Forest       | Many decision trees voting together for a prediction    |
| Training data       | The 25 student records the model learns patterns from   |
| Feature             | An input variable (e.g., MathScore, StudyHours)         |
| Target              | What we're predicting (QuizScore)                       |
| MAE (Mean Abs. Error)| How many points off the model's predictions are on avg |
| StandardScaler      | Normalizes all features to the same scale (0 mean)      |
| Threshold           | Minimum score considered "acceptable" (default: 60)     |

---

## 🙌 Acknowledgements

Built with:
- [Streamlit](https://streamlit.io) — for the interactive web dashboard
- [scikit-learn](https://scikit-learn.org) — for the Random Forest model
- [pandas](https://pandas.pydata.org) — for data loading and manipulation
- [matplotlib](https://matplotlib.org) — for the performance bar chart

---

*SmartStudy AI — Study smarter, not just harder.* 🎓
