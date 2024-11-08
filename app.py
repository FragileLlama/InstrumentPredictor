# app.py

import streamlit as st
import numpy as np
import librosa
import pickle
import soundfile as sf

def extract_features(audio_data, sample_rate):
    try:
        #extract MFCC features
        mfccs = np.mean(librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40).T, axis=0)
        return mfccs
    except Exception as e:
        st.write(f"Error extracting features: {e}")
        return None

def predict_instrument(audio_data, sample_rate):
    #load the trained model
    with open('instrument_classifier.pkl', 'rb') as f:
        clf = pickle.load(f)

    #load the label encoder
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)

    #extract features
    features = extract_features(audio_data, sample_rate)
    if features is not None:
        features = features.reshape(1, -1)
        prediction = clf.predict(features)
        instrument = le.inverse_transform(prediction)
        return instrument[0]
    else:
        return "Could not extract features from the audio."

def main():
    st.title("Audio-Based Instrument Classification")
    st.write("Upload an audio file (MP3 or WAV) to identify the primary instrument.")

    uploaded_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3'])

    if uploaded_file is not None:
        #load the audio file
        try:
            #read the audio file
            audio_data, sample_rate = librosa.load(uploaded_file, sr=None)
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=1)
            instrument = predict_instrument(audio_data, sample_rate)
            st.write(f"The primary instrument is: **{instrument}**")
        except Exception as e:
            st.write(f"Error processing audio file: {e}")

if __name__ == "__main__":
    main()
