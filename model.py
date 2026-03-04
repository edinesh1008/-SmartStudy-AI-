# =============================================================================
# model.py — SmartStudy AI: Core ML Model & Recommendation Engine
# =============================================================================
# This file handles:
#   1. Loading student performance data
#   2. Training a Random Forest model to predict quiz scores
#   3. Identifying each student's weakest subject
#   4. Generating personalized study recommendations
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings("ignore")  # Keep output clean for beginners

# ------------------------------------------------------------------
# STEP 1: Load the dataset
# ------------------------------------------------------------------
def load_data(filepath="student_data.csv"):
    """
    Reads the CSV file into a pandas DataFrame.
    Returns the full DataFrame for inspection and training.
    """
    df = pd.read_csv(filepath)
    print(f"✅ Loaded dataset with {len(df)} student records.")
    print(df.head())
    return df


# ------------------------------------------------------------------
# STEP 2: Train the Random Forest model
# ------------------------------------------------------------------
def train_model(df):
    """
    Trains a Random Forest Regressor to predict QuizScore
    from subject scores and study hours.

    Returns:
        model   — the trained Random Forest model
        scaler  — the fitted StandardScaler (to normalize future inputs)
        mae     — Mean Absolute Error on test data (how accurate the model is)
    """
    # Features (inputs) and target (what we want to predict)
    FEATURES = ["MathScore", "PhysicsScore", "ProgrammingScore", "StudyHours"]
    TARGET   = "QuizScore"

    X = df[FEATURES]
    y = df[TARGET]

    # Split into 80% training data and 20% test data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Normalize features so all scores are on the same scale
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # Train Random Forest (100 decision trees working together)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate accuracy
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"\n🎯 Model trained! Mean Absolute Error: {mae:.2f} points")

    return model, scaler, mae


# ------------------------------------------------------------------
# STEP 3: Predict performance for a new student
# ------------------------------------------------------------------
def predict_performance(model, scaler, math, physics, programming, study_hours):
    """
    Given a student's subject scores and study hours,
    predicts their expected QuizScore.

    Parameters:
        model         — trained Random Forest model
        scaler        — fitted StandardScaler
        math          — Math score (0–100)
        physics       — Physics score (0–100)
        programming   — Programming score (0–100)
        study_hours   — hours studied per day

    Returns:
        predicted_score — float, the predicted quiz score
    """
    input_data = np.array([[math, physics, programming, study_hours]])
    input_scaled = scaler.transform(input_data)
    predicted_score = model.predict(input_scaled)[0]
    return round(predicted_score, 2)


# ------------------------------------------------------------------
# STEP 4: Identify the weakest subject
# ------------------------------------------------------------------
def identify_weak_subjects(math, physics, programming, threshold=60):
    """
    Compares scores for each subject against a threshold.
    Any subject below the threshold is flagged as 'weak'.

    Parameters:
        math, physics, programming — subject scores
        threshold — minimum acceptable score (default: 60)

    Returns:
        weak_subjects — list of subject names that need improvement
        scores_dict   — dict of {subject: score} for charting
    """
    scores_dict = {
        "Math":        math,
        "Physics":     physics,
        "Programming": programming,
    }

    # Find all subjects below the threshold
    weak_subjects = [
        subject for subject, score in scores_dict.items()
        if score < threshold
    ]

    # If everything is above threshold, flag the single lowest subject
    if not weak_subjects:
        weakest = min(scores_dict, key=scores_dict.get)
        weak_subjects = [weakest]  # Still recommend improvement for lowest score

    return weak_subjects, scores_dict


# ------------------------------------------------------------------
# STEP 5: Generate personalized study recommendations
# ------------------------------------------------------------------

# Topic bank — maps each subject to focused study tips
STUDY_TIPS = {
    "Math": [
        "📐 Review algebra fundamentals: equations, inequalities, and functions.",
        "📊 Practice solving 10 problems daily from weak chapters.",
        "🔢 Focus on number theory, ratios, and percentages.",
        "📈 Use Khan Academy's Math section for free guided lessons.",
        "⏱️  Spend at least 45 minutes daily on Math problem sets.",
    ],
    "Physics": [
        "⚡ Revisit Newton's Laws of Motion with real-world examples.",
        "🌊 Study wave mechanics: frequency, amplitude, and wavelength.",
        "🔭 Practice numericals on kinematics and projectile motion.",
        "💡 Watch 3Blue1Brown or Physics Girl on YouTube for visual learning.",
        "📝 Create a formula sheet and review it every morning.",
    ],
    "Programming": [
        "💻 Practice at least one coding problem daily on LeetCode or HackerRank.",
        "🐍 Strengthen Python basics: loops, functions, and data structures.",
        "🔧 Build a small project (calculator, to-do app) to apply concepts.",
        "📖 Read clean code examples on GeeksforGeeks or W3Schools.",
        "🧪 Focus on debugging skills — learn to read error messages carefully.",
    ],
}

def generate_recommendations(weak_subjects):
    """
    Produces a list of study recommendations for each weak subject.

    Parameters:
        weak_subjects — list of subject names (e.g., ['Math', 'Physics'])

    Returns:
        recommendations — dict of {subject: [list of tips]}
    """
    recommendations = {}
    for subject in weak_subjects:
        recommendations[subject] = STUDY_TIPS.get(subject, ["📚 Review your class notes and textbooks."])
    return recommendations


# ------------------------------------------------------------------
# STEP 6: Compute improvement potential
# ------------------------------------------------------------------
def compute_improvement_potential(math, physics, programming, study_hours):
    """
    Estimates how much a student's score could improve
    if they increase their daily study hours by 2.

    Returns a simple percentage-based improvement estimate.
    """
    avg_score       = (math + physics + programming) / 3
    base_potential  = (100 - avg_score) * 0.3   # 30% of remaining gap is achievable
    hour_boost      = min(study_hours * 2, 20)   # More hours = more boost (capped)
    improvement     = round(base_potential + hour_boost, 1)
    return improvement


# ------------------------------------------------------------------
# MAIN — Run a demo when executing this file directly
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 55)
    print("   SmartStudy AI — Recommendation Engine Demo")
    print("=" * 55)

    # Load data and train model
    df             = load_data("student_data.csv")
    model, scaler, mae = train_model(df)

    # Demo: predict for a sample student
    math, physics, programming, study_hours = 55, 48, 70, 3

    print(f"\n📋 Student Input:")
    print(f"   Math: {math}  |  Physics: {physics}  |  Programming: {programming}  |  Study Hours: {study_hours}")

    predicted = predict_performance(model, scaler, math, physics, programming, study_hours)
    print(f"\n🔮 Predicted Quiz Score: {predicted}")

    weak_subjects, scores = identify_weak_subjects(math, physics, programming)
    print(f"\n⚠️  Weak Subjects Detected: {', '.join(weak_subjects)}")

    recs = generate_recommendations(weak_subjects)
    print("\n📚 Personalized Study Recommendations:")
    for subject, tips in recs.items():
        print(f"\n  [{subject}]")
        for tip in tips:
            print(f"    {tip}")

    improvement = compute_improvement_potential(math, physics, programming, study_hours)
    print(f"\n🚀 Improvement Potential: +{improvement} points with consistent effort!")
    print("=" * 55)
