import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from flask import Flask, render_template, request
import google.generativeai as genai
import os

# Initialize Flask app
app = Flask(__name__, template_folder='templates')

# Load dataset
data = pd.read_csv("StressLevelDataset.csv")
encoder = LabelEncoder()
data["stress_level"] = encoder.fit_transform(data["stress_level"])

# Split dataset into features and target
X = data.drop("stress_level", axis=1)
y = data["stress_level"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a decision tree classifier
tree_clf = DecisionTreeClassifier(max_depth=7, random_state=100)
tree_clf.fit(X_train, y_train)

# Set up Gemini API with a valid key
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyAHwYOOvw5VDNRuRf_OxXILMXfQ-1iozzM")  # Ensure it's set
genai.configure(api_key=GOOGLE_API_KEY)

def get_gemini_suggestions(stress_level):
    """Generate stress-relief tips, yoga positions, and relaxation techniques."""
    prompt = f"""Give only 5 points without any markdown
    I am feeling {stress_level} stress level. 
    Suggest 3 effective stress-relief techniques, 2 yoga positions, 
    and 2 relaxation exercises to help me.
    """
    try:
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(prompt)
        return response.text if response else "No suggestions available."
    except Exception as e:
        return "⚠️ Error fetching suggestions. Please check API key."

def get_spotify_link(stress_level):
    """Return Spotify playlist link based on stress level."""
    spotify_links = {
        "0": "https://open.spotify.com/playlist/4BME5NDpshjSW4Gxsnpyul",
        "1": "https://open.spotify.com/playlist/37i9dQZF1DX5yXx6e61fbM",
        "2": "https://open.spotify.com/playlist/6FLheySEeALHtKK51eQGxU"
    }
    return spotify_links.get(str(stress_level), spotify_links[str(stress_level)])

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        try:
            # Get user input
            features = [float(request.form.get(key)) for key in [
                'anxiety_level', 'mental_health_history', 'depression', 'headache', 
                'sleep_quality', 'breathing_problem', 'living_conditions', 
                'academic_performance', 'study_load', 'future_career_concerns', 
                'extracurricular_activities'
            ]]

            # Predict stress level
            user_input = np.array([features])
            predicted_stress_level = tree_clf.predict(user_input)[0]
            predicted_stress_level = encoder.inverse_transform([predicted_stress_level])[0]

            # Get AI-generated tips
            suggestions = get_gemini_suggestions(predicted_stress_level)

            # Get Spotify playlist link
            spotify_link = get_spotify_link(predicted_stress_level)

            return render_template('result.html', 
                                   stress_level=predicted_stress_level,
                                   suggestions=suggestions,
                                   spotify_link=spotify_link)
        except ValueError:
            return render_template('error.html', error_message="Invalid input. Please enter numeric values.")

    return render_template('login.html')

if __name__ == '__main__':
    app.run(debug=True)
