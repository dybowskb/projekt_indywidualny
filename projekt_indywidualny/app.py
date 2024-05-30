import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import os
import pandas as pd
import librosa

app = Flask(__name__)

# Załadowanie modelu
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Mapa wyników predykcji na nazwy gatunków
genre_map = {0: 'Dance', 1: 'Muzyka Klasyczna', 2: 'Rock'}

def extract_features(file_name):
    y, sr = librosa.load(file_name)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)

    # Wyodrębnienie cech
    tempo = float(librosa.beat.tempo(onset_envelope=onset_env, sr=sr))
    #tempo = float(tempo_array[0])  # Wybór pierwszej wartości tempa i konwersja na float
    key, _ = librosa.beat.beat_track(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y).mean()
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    #spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
    #spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr).mean()
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
    spectral_flatness = librosa.feature.spectral_flatness(y=y).mean()
    rms = librosa.feature.rms(y=y).mean()
    harmonic_to_noise = librosa.effects.harmonic(y).mean()
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr).mean()
    #chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr).mean()
    mfcc = librosa.feature.mfcc(y=y, sr=sr).mean()
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    #rhythmics = librosa.feature.tempogram(y=y, sr=sr).mean()

    return np.array([tempo, key.mean(), zero_crossing_rate, spectral_centroid,
                     spectral_rolloff, spectral_flatness, rms, harmonic_to_noise,
                     chroma_stft,  mfcc, pitches.mean()])


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']

    if file.filename == '':
        return "No selected file", 400

    try:
        print("File received:", file.filename)
        file_path = os.path.join("temp", file.filename)
        file.save(file_path)
        print("File saved at:", file_path)

        features = extract_features(file_path)
        print("Features extracted:", features)

        features_df = pd.DataFrame([features], columns=['tempo', 'key', 'zero_crossing_rate', 'spectral_centroid',
                                                        'spectral_rolloff', 'spectral_flatness', 'rms', 'harmonic_to_noise',
                                                        'chroma_stft', 'mfcc', 'pitches'])

        prediction = model.predict(features_df)
        genre = genre_map[int(prediction[0])]
        print("Prediction made:", genre)

        # Usunięcie tymczasowego pliku
        os.remove(file_path)

        return jsonify({'prediction': genre})

    except Exception as e:
        print("Error during prediction:", e)
        return str(e), 500


if __name__ == "__main__":
    if not os.path.exists("temp"):
        os.makedirs("temp")
    app.run(debug=True)

