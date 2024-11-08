# predict.py

import numpy as np
import librosa
import pickle
import sys

def extract_features(file_name):
    try:
        audio_data, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        #extract MFCC features
        mfccs = np.mean(librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40).T, axis=0)
        return mfccs
    except Exception as e:
        print(f"Error parsing file {file_name}: {e}")
        return None

def predict_instrument(file_name):
    #load the trained model
    with open('instrument_classifier.pkl', 'rb') as f:
        clf = pickle.load(f)

    #load the label encoder
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)

    #extract features
    features = extract_features(file_name)
    if features is not None:
        features = features.reshape(1, -1)
        prediction = clf.predict(features)
        instrument = le.inverse_transform(prediction)
        print(f"The primary instrument is: {instrument[0]}")
    else:
        print("Could not extract features from the audio.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_name = sys.argv[1]
        predict_instrument(file_name)
    else:
        print("Please provide an audio file path.")
