import joblib
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split  # Add this import
import librosa
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to extract features from audio files
def extract_features(audio_file):
    try:
        y, sr = librosa.load(audio_file)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=42)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        chroma_features = librosa.feature.chroma_stft(y=y, sr=sr)
        rms = librosa.feature.rms(y=y)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        features = np.hstack([
            np.mean(mfccs, axis=1), np.std(mfccs, axis=1),
            np.mean(spectral_centroid), np.std(spectral_centroid),
            np.mean(zero_crossing_rate), np.std(zero_crossing_rate),
            np.mean(spectral_bandwidth), np.std(spectral_bandwidth),
            np.mean(spectral_contrast, axis=1), np.std(spectral_contrast, axis=1),
            np.mean(chroma_features, axis=1), np.std(chroma_features, axis=1),
            np.mean(rms), np.std(rms),
            np.mean(spectral_rolloff), np.std(spectral_rolloff),
            np.mean(mel_spectrogram, axis=1), np.std(mel_spectrogram, axis=1)
        ])
        return features
    except Exception as e:
        logging.error(f"Error extracting features from {audio_file}: {e}")
        return None

# Function to preprocess data
def preprocess_data(base_dirs, is_test=False):
    all_features = []
    labels = []

    for base_dir in base_dirs:
        bird_names = os.listdir(base_dir)
        
        for bird_name in bird_names:
            bird_dir = os.path.join(base_dir, bird_name)
            for f in os.listdir(bird_dir):
                features = extract_features(os.path.join(bird_dir, f))
                if features is not None:
                    all_features.append(features)
                    if is_test:
                        labels.append(bird_name.replace('_test', ''))
                    else:
                        labels.append(bird_name)

    X = np.array(all_features)
    y = np.array(labels)
    
    return X, y

# Function to load data, train the model, and return the trained model, scaler, and PCA
def load_and_train_model():
    base_dirs = ['critically_endangered', 'endangered', 'vulnerable']
    X, y = preprocess_data(base_dirs)

    # Shuffle the data
    shuffled_indices = np.random.RandomState(42).permutation(len(y))
    X = X[shuffled_indices]
    y = y[shuffled_indices]

    # Fit the scaler on all collected features
    scaler = StandardScaler()
    scaler.fit(X)

    # Normalize features using the fitted scaler
    X_scaled = scaler.transform(X)

    # Dimensionality reduction (optional)
    n_components = min(X_scaled.shape[0], X_scaled.shape[1])
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

    # Define classifiers
    classifiers = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'k-NN': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'SVM': SVC(kernel='linear', random_state=42),
        'MLP': MLPClassifier(random_state=42),
        'Naive Bayes': GaussianNB(),
        'Logistic Regression': LogisticRegression(random_state=42)
    }

    # Train the best classifier
    best_classifier_name = 'Logistic Regression'
    classifier = classifiers[best_classifier_name]
    classifier.fit(X_train, y_train)

    return classifier, scaler, pca

# Function to predict bird species from an audio file
def predict_bird_species(audio_file, classifier, scaler, pca):
    features = extract_features(audio_file)
    if features is None:
        return "Error: Could not extract features from the audio file."

    # Normalize and apply PCA transformation
    features_scaled = scaler.transform([features])
    features_pca = pca.transform(features_scaled)

    # Predict the bird species
    prediction = classifier.predict(features_pca)
    return prediction[0]
  
