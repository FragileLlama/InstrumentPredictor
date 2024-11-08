# data_preparation.py

import os
import librosa
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import json
import pickle
from tqdm import tqdm  #progress bar

def extract_features(file_name):
    try:
        audio_data, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        #extract MFCC features
        mfccs = np.mean(librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40).T, axis=0)
        return mfccs
    except Exception as e:
        print(f"Error parsing file {file_name}: {e}")
        return None

def prepare_dataset():
    dataset_path = 'nsynth-test/'
    audio_path = os.path.join(dataset_path, 'audio')
    metadata_path = os.path.join(dataset_path, 'examples.json')

    #load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    features = []
    labels = []
    file_names = []

    print("Extracting features from audio files...")

    #iterate through each audio file in the audio directory
    for key in tqdm(metadata.keys()):
        file_name = key + '.wav'
        file_path = os.path.join(audio_path, file_name)

        if os.path.exists(file_path):
            data = extract_features(file_path)
            if data is not None:
                features.append(data)
                label = metadata[key]['instrument_family_str']
                labels.append(label)
                file_names.append(file_name)
        else:
            print(f"File {file_name} does not exist.")

    #convert to DataFrame
    features_df = pd.DataFrame(features)
    features_df['label'] = labels
    features_df['file_name'] = file_names

    #encode labels
    le = LabelEncoder()
    features_df['label_encoded'] = le.fit_transform(features_df['label'])

    #save Label Encoder
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)

    #save features to CSV
    features_df.to_csv('features.csv', index=False)
    print("Dataset prepared and saved to features.csv")

if __name__ == "__main__":
    prepare_dataset()
