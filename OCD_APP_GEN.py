from flask import Flask, render_template_string, request
import pandas as pd
import joblib
from openai import OpenAI
from dotenv import load_dotenv
import os

#TO RUN IN TERMINAL:
#cd "/Users/josephine/Desktop/untitled folder"
#source venv/bin/activate
#pip3 install flask openai python-dotenv
#python3 OCD_APP_GEN.py

#Load model and features, encode the target
directory = "models"
rf_model = joblib.load(f"{directory}/rf.joblib")
rf_features = joblib.load(f"{directory}/features.joblib")

load_dotenv("key.env") #load key for OPENAI API
client = OpenAI()

#instructions for AI
INSTRUCTIONS = (
    "You are the research assistant to Psychiatrist. You will receive patient information and a drug GROUP (SSRI, SNRI, or Benzodiazepine) that has been determined by a clinical model."
    "Your task is to select 2-3 specific medications ONLY from the given drug group, using the patient's profile to choose the most appropriate options within that group." 
    "Do not question or change the drug group - it has been determined by the model. Explain your reasoning for selecting these specific medications based on the patient's characteristics and keep in mind that the main issue is OCD." 
    "Use concise and scientific-sounding speech. Mention that the generation is based on the patient profile and predictive model, not only the patient profile."
    "Precise that this is only a recommendation and needs to be approved by a doctor."
)

def AI_generator(drug_group: str, features: dict) -> str:
#patient info to be passed into AI - making it more readable for generation
    patient_info = f"""
        Patient Profile:
        - Age: {features['Age']} years
        - Duration of Symptoms: {features['Duration of Symptoms (months)']} months
        - Y-BOCS Obsessions Score: {features['Y-BOCS Score (Obsessions)']}/40
        - Y-BOCS Compulsions Score: {features['Y-BOCS Score (Compulsions)']}/40
        - Depression: {'Yes' if features['Depression'] == 1 else 'No'}
        - Anxiety: {'Yes' if features['Anxiety'] == 1 else 'No'}
        """
    #generation happens here
    generation = client.chat.completions.create(
        #what model of gpt
        model="gpt-4o", #what model of gpt
        #instructions, drug group, and patient info passed to AI to help generate
        messages=[ 
            {"role": "system", "content": INSTRUCTIONS},
            {"role": "user", "content": f"{patient_info}\nDrug group: {drug_group}"}
        ],
        #helps with randomness 
        temperature=0.4
    )
    #return generation
    return generation.choices[0].message.content.strip()

#function to pass new patientsfeatures into the saved random forest model and predict
def predict_drug_class(features: dict) -> str:
    X = pd.DataFrame([features])[rf_features]
    prediction = rf_model.predict(X)[0]
    return prediction  

app = Flask(__name__) #flask app

#html to design the web app 
FORM_HTML = '''
<!doctype html>
<body style="background-color: #5f9ccf; color: white; padding: 20px; font-size: 18px;">
<title>OCD Treatment Recommender - Psychiatrist Assistant</title>
<div style="text-align: center;">
  <h2>OCD Treatment Recommender - Psychiatrist Assistant</h2>
  <form method=post>
    Duration of OCD symptoms (months): <input type=number name=duration required style="width: 50px;"><br><br>
    Age: <input type=number name=age required style="width: 50px;"><br><br>
    Y-BOCS Obsessions score (0-40): <input type=number name=obs min=0 max=40 required style="width: 50px;"><br><br>
    Y-BOCS Compulsions score (0-40): <input type=number name=comp min=0 max=40 required style="width:50px;"><br><br>
    Depression (0=No, 1=Yes): <select name=dep style="width: 50px;"><option value=0>0</option><option value=1>1</option></select><br><br>
    Anxiety (0=No, 1=Yes): <select name=anx style="width: 50px;"><option value=0>0</option><option value=1>1</option></select><br><br>
    <input type=submit value="Get Recommendations">
  </form>
</div>
{% if result %}
  <div style="text-align: left; margin-top: 20px;">
    <b>Predicted Drug Group:</b> {{ result['group'] }}<br><br>
    <b>Recommendations:</b><br>
    <pre style="white-space: pre-wrap; word-wrap: break-word; max-width: 95%;">{{ result['recommendations'] }}</pre>
  </div>
{% endif %}
</body>
'''

@app.route('/', methods=['GET', 'POST']) #web app - allowed to view and submit things
def index():
    result = None
    if request.method == 'POST': #if user pressed button
        try:
            features = { #get data from user input
                "Duration of Symptoms (months)": float(request.form['duration']),
                "Age": float(request.form['age']),
                "Y-BOCS Score (Obsessions)": float(request.form['obs']),
                "Y-BOCS Score (Compulsions)": float(request.form['comp']),
                "Depression": int(request.form['dep']),
                "Anxiety": int(request.form['anx'])
            }
            drug_group = predict_drug_class(features) #run prediction
            generation = AI_generator(drug_group, features) #run generation
            result = {'group': drug_group, 'recommendations': generation} 
        except:
            result = {'group': 'Error', 'recommendations': 'Error'} #for errors
    return render_template_string(FORM_HTML, result=result) #html

if __name__ == '__main__':
    app.run(debug=True, port=5001) #run app