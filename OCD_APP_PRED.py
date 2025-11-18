# Josephine Loiret-Bernal
# CSEN 166 - Personal Project

#TO RUN IN TERMINAL:
#cd "/Users/josephine/Desktop/untitled folder"
#python3 -m venv venv
#source venv/bin/activate
#pip3 install pandas scikit-learn joblib
#python3 OCD_APP_PRED.py


#import libraries and dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

filepath = 'ocd_patient_dataset.csv'

df = pd.read_csv(filepath)
print(df.head())

#pre-processing
df = df.dropna(subset=['Medications']) #remove where medication is None
df["Depression"] = df["Depression Diagnosis"].map({"Yes": 1, "No": 0})
df["Anxiety"] = df["Anxiety Diagnosis"].map({"Yes": 1, "No": 0})
features = ['Duration of Symptoms (months)', 'Age', 'Y-BOCS Score (Obsessions)', 'Y-BOCS Score (Compulsions)', 'Depression', 'Anxiety']
X = df[features] #features I want to use to train model
y = df["Medications"]

#split 70% train data - 30% test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#training predictive model - Random Forest Model - predict drug group
rf = RandomForestClassifier(n_estimators=100, max_depth=3, oob_score=True, random_state=42,)
rf.fit(X_train, y_train)
print(f"OOB Score: {rf.oob_score_:.4f}")

#predictions, look at accuracy, check for overfitting
train_predictions = rf.predict(X_train)
test_predictions = rf.predict(X_test)
print("Training Accuracy:", accuracy_score(y_train, train_predictions))
print("Testing Accuracy:", accuracy_score(y_test, test_predictions))
print()
print(classification_report(y_test, test_predictions))
print()
print(confusion_matrix(y_test, test_predictions))

#save model so we can use it with pretrained generative model
import joblib, os
os.makedirs("models", exist_ok=True)     # save next to the notebook
joblib.dump(rf, "models/rf.joblib")
joblib.dump(features, "models/features.joblib") 