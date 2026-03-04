import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
data = pd.read_csv("student_data.csv")

# Features
X = data[['MathScore','PhysicsScore','ProgrammingScore','StudyHours']]

# Target
y = data['QuizScore']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

# Train model
model = RandomForestRegressor()
model.fit(X_train,y_train)

# Save trained model
joblib.dump(model,"smartstudy_model.pkl")

print("Model trained and saved successfully")