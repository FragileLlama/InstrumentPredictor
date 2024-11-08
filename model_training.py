# model_training.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

def train_model():
    #load features
    data = pd.read_csv('features.csv')
    X = data.drop(['label', 'label_encoded', 'file_name'], axis=1)
    y = data['label_encoded']

    #split into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #initialize classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)

    #train the classifier
    clf.fit(X_train, y_train)

    #evaluate the model
    score = clf.score(X_test, y_test)
    print(f"Model accuracy: {score * 100:.2f}%")

    #save the model
    with open('instrument_classifier.pkl', 'wb') as f:
        pickle.dump(clf, f)
    print("Model saved as instrument_classifier.pkl")

if __name__ == "__main__":
    train_model()
