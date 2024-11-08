I developed an Audio-Based Instrument Classification System designed to identify the primary instrument in an audio file. Using Python and libraries like librosa for audio processing and scikit-learn for machine learning, I built a model that can classify instruments based on sound features. The system allows users to upload an MP3 or WAV file through a user-friendly interface created with Streamlit. Upon uploading, the audio is processed to extract features such as Mel-frequency cepstral coefficients (MFCCs), which are then fed into a trained Random Forest Classifier to predict the instrument. I trained the model on the NSynth Dataset, which is a large collection of annotated audio samples from various instruments.

The project involves several key steps: data preparation, where I extracted audio features and encoded labels; model training, where the classifier learns to distinguish between different instruments; and deployment, where I integrated the model into a web app for user interaction. Below are examples of the system successfully identifying a keyboard sample and a guitar sample:



How to Run the Project
Prerequisites
Python 3.7 or later
Git
pip (Python package installer)
NSynth Dataset: Download the NSynth Test Set and extract it into the project directory.
Steps to Run the Project
Clone the Repository

Open your terminal or command prompt and run:

bash
Copy code
git clone https://github.com/yourusername/audio-instrument-classification.git
Replace yourusername with your actual GitHub username.

Navigate to the Project Directory

bash
Copy code
cd audio-instrument-classification
Set Up a Virtual Environment (Optional but Recommended)

Create a virtual environment:

bash
Copy code
python -m venv venv
Activate the virtual environment:

On Windows:

bash
Copy code
venv\Scripts\activate
On macOS/Linux:

bash
Copy code
source venv/bin/activate
Install Required Dependencies

bash
Copy code
pip install -r requirements.txt
Download and Prepare the NSynth Dataset

Download the NSynth Test Set from here.
Extract the dataset into the project directory so that you have a folder named nsynth-test containing audio/ and examples.json.
Run the Data Preparation Script

bash
Copy code
python data_preparation.py
This script extracts features from the audio files and saves them to features.csv.

Train the Model

bash
Copy code
python model_training.py
This will train the Random Forest Classifier and save the model as instrument_classifier.pkl.

Run the Streamlit Application

bash
Copy code
streamlit run app.py
This will launch the web application. Open your web browser and navigate to http://localhost:8501.

Upload an Audio File

Use the web interface to upload an MP3 or WAV file.
The application will display the predicted primary instrument.
Future Development
For future development, I plan to enhance the model's accuracy by incorporating deep learning techniques such as Convolutional Neural Networks (CNNs) or Recurrent Neural Networks (RNNs). Expanding the dataset to include more diverse instruments and genres could also improve performance. Additionally, I'm considering developing the capability to recognize multiple instruments within a single audio file and adding features like real-time audio processing and visualization of audio spectrograms to make the tool more versatile and user-friendly.

Feel free to explore the project and contribute! If you encounter any issues or have suggestions, please open an issue or submit a pull request.






