import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# 1. Load the dataset
df = pd.read_csv('student_data.csv')

# 2. Separate features (X) and target (y)
X = df[['12th_Marks', 'Maths_Score', 'CS_Score']]
y = df['Final_Performance']

# 3. Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize and train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Test the model and print accuracy
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model trained successfully! Accuracy: {accuracy * 100:.2f}%")

# 6. Save the trained model to a file using pickle
with open('edupredict_model.pkl', 'wb') as file:
    pickle.dump(model, file)
print("Model saved as 'edupredict_model.pkl'")