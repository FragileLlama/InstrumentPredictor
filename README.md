# Audio-Based Instrument Classification System

I developed an **Audio-Based Instrument Classification System** designed to identify the primary instrument in an audio file. Using Python and libraries like **librosa** for audio processing and **scikit-learn** for machine learning, I built a model that classifies instruments based on sound features. The system allows users to upload an MP3 or WAV file through a user-friendly interface created with **Streamlit**. Upon uploading, the audio is processed to extract features such as **Mel-frequency cepstral coefficients (MFCCs)**, which are then fed into a trained **Random Forest Classifier** to predict the instrument. The model is trained on the **NSynth Dataset**, a large collection of annotated audio samples from various instruments.

## Project Overview
The project involves several key steps:
- **Data Preparation**: Extracting audio features and encoding labels.
- **Model Training**: Training a classifier to distinguish between different instruments.
- **Deployment**: Integrating the model into a web app for user interaction.

### Examples
Below are examples of the system successfully identifying a keyboard sample and a guitar sample.

---

## How to Run the Project

### Prerequisites
- Python 3.7 or later
- Git
- pip (Python package installer)
- NSynth Dataset: Download the **NSynth Test Set** and extract it into the project directory.

### Steps to Run the Project

#### 1. Clone the Repository
Open your terminal or command prompt and run:

```bash
git clone https://github.com/FragileLlama/audio-instrument-classification.git
```

#### 2. Navigate to the Project Directory
```bash
cd InstrumentPredictor
```

#### 3. Download and Prepare the NSynth Dataset
### 1. Download the NSynth Test Set from Google Research’s NSynth Dataset page.
### 2. Extract the dataset into the project directory so that you have a folder named nsynth-test containing audio/ and examples.json.

#### 4. Run the scripts:

```bash
python data_preparation.py
python model_training.py
streamlit run app.py
```

#### 5. Upload an Audio File
Use the web interface to upload an MP3 or WAV file. The application will display the predicted primary instrument.






