import streamlit as st
import librosa
import numpy as np
import joblib
from pydub import AudioSegment
import xgboost
import matplotlib.pyplot as plt
import shutil
import os

# Function to check if ffmpeg is installed
def is_ffmpeg_installed():
    try:
        shutil.which("ffmpeg")
        return True
    except FileNotFoundError:
        return False

# Function to extract features from an audio file
def extract_features(audio_path):
    # Load audio file
    y, sr = librosa.load(audio_path, sr=None)

    # Extract relevant features using librosa
    features = [
        librosa.feature.zero_crossing_rate(y),
        librosa.feature.spectral_centroid(y=y, sr=sr),
        librosa.feature.spectral_bandwidth(y=y, sr=sr),
        librosa.feature.spectral_rolloff(y=y, sr=sr),
        librosa.feature.mfcc(y=y, sr=sr),
    ]

    # Flatten the features into a 1D array
    flat_features = np.concatenate([feature.mean(axis=1) for feature in features])

    # Keep only the first 18 features (these are the features relevant to our model)
    selected_features = flat_features[:18]

    return selected_features

def predict_parkinsons(features):
    # Load the saved model
    loaded_model = joblib.load('xgboost_model.joblib')

    # Reshape the features array
    features = features.reshape((1, -1))

    # Make predictions
    predictions = loaded_model.predict(features)

    return predictions[0]

# Main function
def main():
    st.set_page_config(page_title="Neurogen", page_icon=":smiley:")

    # Check if ffmpeg is installed
    if not is_ffmpeg_installed():
        st.error("Your system shows you do not have ffmpeg installed, you need to have ffmpeg or an alternative installed to run this application.")
        return

    st.title("Parkinson's Prediction App")

    # Navigation
    pages = ["Introduction", "Predictor"]
    
    # Create toggle button for small screens
    toggle_button = """
        <style>
            @media only screen and (max-width: 600px) {
                .stSidebar, .sidebar .dropdown-content, .stButton>button {
                    display: block;
                    width: 100%;
                }
            }
        </style>
    """
    st.markdown(toggle_button, unsafe_allow_html=True)
    
    page_selection = st.sidebar.radio("Navigation", pages)

    if page_selection == "Introduction":
        st.header("Welcome to Parkinson's Prediction App")
        st.write(
            "This app allows you to upload an audio file, extract relevant features, "
            "and predict the likelihood of Parkinson's disease based on those features. "
            "The app uses AI to make the predictions with an average of ~94.8% accuracy. "
            "This app is not a replacement for medical opinion, for further assistance on Parkinsons,\nkindly speak to a doctor."
        )
        st.write(
            "To get started, click on 'Predictor' in the navigation bar on the left and upload your audio file. "
            "Click the arrow icon (on your extreme left corner) to toggle the menu on small screens."
        )

    elif page_selection == "Predictor":
        # Predictions
        st.header("Voice Prediction")

        # Upload audio file
        uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

        if uploaded_file is not None:
            # Convert the audio file to WAV format
            audio = AudioSegment.from_file(uploaded_file)
            audio_path = "uploaded_audio.wav"
            audio.export(audio_path, format="wav")

            # Extract features from the converted audio file
            extracted_features = extract_features(audio_path)

            # Make predictions based on the extracted features
            prediction = predict_parkinsons(extracted_features)

            # Display prediction
            st.subheader("Prediction:")
            if prediction == 1:
                st.error("High likelihood of Parkinson's disease. Please consult with a healthcare professional.")
            else:
                st.success("Low likelihood of Parkinson's disease. Regular monitoring is still recommended.")

            # Visualization of extracted features
            st.subheader("Extracted Features Visualization")

            # Plotting the bar chart
            feature_names = [
                "Zero Crossing Rate",
                "Spectral Centroid",
                "Spectral Bandwidth",
                "Spectral Rolloff",
                "MFCC 1",
                "MFCC 2",
                "MFCC 3",
                "MFCC 4",
                "MFCC 5",
                "MFCC 6",
                "MFCC 7",
                "MFCC 8",
                "MFCC 9",
                "MFCC 10",
                "MFCC 11",
                "MFCC 12",
                "MFCC 13",
                "MFCC 14",
            ]

            plt.figure(figsize=(10, 6))
            plt.bar(feature_names, extracted_features)
            plt.xlabel("Feature")
            plt.ylabel("Value")
            plt.title("Extracted Features")
            st.pyplot(plt)

            # Allow users to upload another file
            if st.button("Upload Another File"):
                st.file_uploader("Upload an audio file", type=["wav", "mp3"])

if __name__ == "__main__":
    main()
