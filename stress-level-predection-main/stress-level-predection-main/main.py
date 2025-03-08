import numpy as np
import pandas as pd
import streamlit as st
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import google.generativeai as genai
import os

# Set up API key
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyAHwYOOvw5VDNRuRf_OxXILMXfQ-1iozzM")
genai.configure(api_key=GOOGLE_API_KEY)

# Load dataset
data = pd.read_csv("StressLevelDataset.csv")
encoder = LabelEncoder()
data["stress_level"] = encoder.fit_transform(data["stress_level"])

# Prepare data
X = data.drop("stress_level", axis=1)
y = data["stress_level"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree model
tree_clf = DecisionTreeClassifier(max_depth=7, random_state=100)
tree_clf.fit(X_train, y_train)

# Streamlit UI
st.title("🧘 Stress Level Prediction & AI-Based Suggestions")

# User Input Fields
st.subheader("🔹 Enter Your Details:")
feature_names = [
    'Anxiety Level', 'Mental Health History', 'Depression', 'Headache', 
    'Sleep Quality', 'Breathing Problem', 'Living Conditions', 
    'Academic Performance', 'Study Load', 'Future Career Concerns', 
    'Extracurricular Activities'
]

user_input = []
for name in feature_names:
    user_input.append(st.number_input(f"{name} (0-10)", min_value=0.0, max_value=10.0, step=0.1))

# Stress Relief Suggestions using Gemini AI
def get_gemini_suggestions(stress_level):
    """Generate stress-relief tips using Gemini AI."""
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

# Spotify Relaxation Playlists
def get_spotify_link(stress_level):
    """Return Spotify playlist link based on stress level."""
    spotify_links = {
        "0": "https://open.spotify.com/playlist/4BME5NDpshjSW4Gxsnpyul",
        "1": "https://open.spotify.com/playlist/37i9dQZF1DX5yXx6e61fbM",
        "2": "https://open.spotify.com/playlist/6FLheySEeALHtKK51eQGxU"
    }
    return spotify_links.get(str(stress_level), spotify_links["0"])

# Prediction
if st.button("🔍 Predict Stress Level"):
    try:
        user_array = np.array([user_input])
        predicted_stress = tree_clf.predict(user_array)[0]
        predicted_stress_label = encoder.inverse_transform([predicted_stress])[0]

        st.success(f"🧠 **Predicted Stress Level:** {predicted_stress_label}")
        
        # Fetch AI-generated suggestions
        with st.spinner("Fetching personalized stress-relief tips..."):
            suggestions = get_gemini_suggestions(predicted_stress_label)
            st.subheader("💡 AI-Generated Stress-Relief Tips:")
            st.write(suggestions)

        # Provide Spotify playlist
        st.subheader("🎵 Relax with Music:")
        st.markdown(f"[Click here to listen](%s)" % get_spotify_link(predicted_stress_label))

    except ValueError:
        st.error("Invalid input. Please enter numeric values.")

