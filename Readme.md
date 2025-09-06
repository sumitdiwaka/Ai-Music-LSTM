AI Music Generation Studio
A full-stack web application that uses an LSTM neural network to compose original musical melodies and convert existing MIDI files into different genres. The platform features a complete user authentication system, a personal dashboard to save and manage generated songs, and an advanced in-browser player with real-time visualization.

📋 Table of Contents
- Description
- Features
- Technology Stack
- Installation & Setup
- How to Use
- Future Scope

📝 Description
This project is a comprehensive web application that leverages a Long Short-Term Memory (LSTM) neural network to learn musical patterns from a large dataset of MIDI files. It serves as a powerful creative tool, allowing users to generate new, original music from scratch or transform their own MIDI files into different styles like Jazz, Rock, or Classical.

The application is built with a Python and Flask backend that handles the AI model execution and user data management, and a modern, interactive frontend built with JavaScript and Tailwind CSS. It features a complete user authentication system, a personal dashboard for each user to track their creations, and an advanced music player with real-time audio visualization.

✨ Features
- Full User Authentication: Secure user registration and login system with profile photo uploads.
- Personal User Dashboard: A dedicated space for users to view their profile, update their details, and see a history of all their generated songs.
- AI Music Generation: Generate completely new and original musical melodies from scratch using a general-purpose AI model.
- MIDI Style Transfer: Upload a MIDI file and convert it into a different genre (e.g., Classical, Jazz, Rock) using specialized AI models.- 
Advanced Music Player:
- In-browser playback of generated MIDI files.
- Controls to change the instrument sound (Piano, FM Synth, etc.).
- Playback speed controls (slow down and speed up).
- A seekable progress bar to jump to any part of the song.
- Real-time Audio Visualizer: A dynamic waveform that animates as the music plays, providing an engaging visual experience.
- Song Management: Users can view their song history and delete songs they no longer need.

🛠️ Technology Stack
Category
Technology

Backend
Python, Flask, Keras (TensorFlow), SQLAlchemy, Flask-Login, Music21

Frontend
HTML5, JavaScript (ES6+), Tailwind CSS, Tone.js

Database
SQLite

AI Model
Long Short-Term Memory (LSTM) Neural Network

🚀 Installation & Setup
Follow these steps to set up and run the project locally.

1. Prerequisites
Python (version 3.9 or higher recommended)
pip (Python package installer)

2. Clone the Repository
git clone <your-repository-url>
cd music-generation-project

3. Set Up a Virtual Environment
It is highly recommended to use a virtual environment.

# Create the virtual environment in the root folder
python -m venv venv

# Activate it
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

4. Install Dependencies
Navigate to the backend folder and install all required Python libraries.

cd backend
pip install -r requirements.txt

5. Download and Organize the Dataset- The AI models need to be trained on a large dataset of MIDI files.
- Download: Go to the Lakh MIDI Dataset and download the "Clean MIDI subset".
- Extract: Unzip the downloaded file. You will get a folder containing many subfolders named after artists.
- Organize for Style Transfer:
- Inside backend/data/midi_files/, create three new folders: classical, jazz, and rock.
- Go through the artist folders you downloaded and copy MIDI files into the appropriate genre folder (e.g., copy files from the "Mozart" folder into your classical folder). Aim for at least 50-100 files per genre.

6. Preprocess the Data
Run the preprocessing script for your main model and for each genre model.

# Make sure you are in the /backend folder
# (This script is not provided but is a necessary step from the project development)
# python preprocess.py general 

# Process each genre
python preprocess.py classical
python preprocess.py jazz
python preprocess.py rock

This will create notes_classical, notes_jazz, etc., files in your backend/model/ directory.

7. Train the AI Models
This is a time-consuming step. It's best to run this overnight.

# Make sure you are in the /backend folder
# (This assumes a general model was also trained)
# python train.py general 

# Train each genre model (use the high-quality version of the script)
python train.py classical
python train.py jazz
python train.py rock

This will create .keras weight files for each model in your backend/model/ directory.

8. Run the Application
Once the models are trained, you are ready to start the server.

# Make sure you are in the /backend folder
# If you are re-running after code changes, delete users.db first
python app.py

The server will start, load all the AI models, and will be accessible at http://127.0.0.1:5000.

📖 How to Use-
- Open the Web App: Navigate to http://127.0.0.1:5000 in your web browser.
- Create an Account: Go to the signup page, fill in your details, and upload a profile picture.
- Log In: Log in with your new credentials. You will be redirected to your personal dashboard.

Navigate to the Generator: From the dashboard, click the "Generate New Music" button.

  Generate Music:
- To create a new song, select an instrument and click "Generate Music".
- To convert an existing song, upload a MIDI file, select a target genre, and click "Convert Style".
- Enjoy: Use the player controls to listen to, change the speed of, and download your new creation!

🔮 Future Scope-
- Advanced Generation Controls: Allow users to select a musical key, tempo, or mood before generation.
- More Genre Models: Train AI models for more genres like Blues, Electronic, or Pop music.
- Cloud Deployment: Deploy the application to a cloud platform (like AWS or Heroku) to make it publicly accessible.