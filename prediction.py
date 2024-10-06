#!/usr/bin/env python
# coding: utf-8

# ### 1. Setup and Imports

# In[1]:


# Import necessary libraries
import os
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ### 2. Feature Extraction Function

# In[2]:


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


# ### 3. Data Preprocessing Function

# In[3]:


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
        
    
    # Combine features and labels
    X = np.array(all_features)
    y = np.array(labels)
    
    return X, y


# # 4. Load and Preprocess Data

# In[4]:


# Base directories containing audio files for each class
base_dirs = ['critically_endangered', 'endangered', 'vulnerable']

# Preprocess data
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


# ### 5. Define and Train Classifiers

# In[5]:


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

# Train and test each classifier
accuracies = {}
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred_test = clf.predict(X_test)
    accuracies[name] = accuracy_score(y_test, y_pred_test)

# Print initial split test accuracies
print("Initial Split Test Accuracies:")
for name, accuracy in accuracies.items():
    print(f"{name}: {accuracy}")

# Plot accuracies
plt.figure(figsize=(10, 6))
plt.barh(range(len(accuracies)), list(accuracies.values()), align='center')
plt.yticks(range(len(accuracies)), list(accuracies.keys()))
plt.xlabel('Accuracy')
plt.title('Classifier Accuracies on Initial Split Test Data')
plt.gca().invert_yaxis()
plt.show()


# In[6]:


# Evaluate classifiers on the new test data
test_accuracies = {}
best_classifier = None
best_accuracy = 0
best_classifier_name = ""

for name, clf in classifiers.items():
    y_pred_test = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_test)
    test_accuracies[name] = accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_classifier = clf
        best_classifier_name = name



# Generate and display confusion matrix for the best classifier
y_pred_best = best_classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best, labels=best_classifier.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_classifier.classes_)

plt.figure(figsize=(20,20))
disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
plt.title(f"Confusion Matrix for {best_classifier_name}")
plt.show()


# In[7]:


np.shape(X_train)


# ### 6. Evaluate on New Test Data

# In[9]:


# Preprocess test data
test_base_dirs = ['critically_endangered_test', 'endangered_test', 'vulnerable_test']
X_test_new, y_test_new = preprocess_data(test_base_dirs, is_test=True)

# Normalize test features using the fitted scaler
X_test_new_scaled = scaler.transform(X_test_new)

# Apply PCA transformation using the fitted PCA model
X_test_new_pca = pca.transform(X_test_new_scaled)

# Evaluate classifiers on the new test data
test_accuracies = {}
best_classifier = None
best_accuracy = 0
best_classifier_name = ""

for name, clf in classifiers.items():
    y_pred_test_new = clf.predict(X_test_new_pca)
    accuracy = accuracy_score(y_test_new, y_pred_test_new)
    test_accuracies[name] = accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_classifier = clf
        best_classifier_name = name

# Print final test accuracies
print("\nFinal Test Accuracies:")
for name, accuracy in test_accuracies.items():
    print(f"{name}: {accuracy}")

# Plot test accuracies
plt.figure(figsize=(10, 6))
plt.barh(range(len(test_accuracies)), list(test_accuracies.values()), align='center')
plt.yticks(range(len(test_accuracies)), list(test_accuracies.keys()))
plt.xlabel('Accuracy')
plt.title('Classifier Test Accuracies on Final Test Data')
plt.gca().invert_yaxis()
plt.show()


# ### 7. Confusion Matrix for Best Classifier

# In[10]:


# Generate and display confusion matrix for the best classifier
y_pred_best = best_classifier.predict(X_test_new_pca)
cm = confusion_matrix(y_test_new, y_pred_best, labels=best_classifier.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_classifier.classes_)

plt.figure(figsize=(20,20))
disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')

plt.title(f"Confusion Matrix for the Best Classifier {best_classifier_name} on Final Test Data")
plt.show()

