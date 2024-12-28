import os

import librosa
import librosa.display
import numpy as np
import pandas as pd

# Paths and constants
AUDIO_FOLDER = "CBU0521DD_stories"
ATTRIBUTE_FILE = "CBU0521DD_stories_attributes.csv"
FEATURES_FILE = "features.csv"


# Feature extraction function
def extract_features(file_path):
    try:
        # Load audio
        y, sr = librosa.load(file_path, sr=None, mono=True, duration=220)  # Load up to 220 s

        # Extract features
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
        spectral_centroid = np.array([np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))])
        spectral_bandwidth = np.array([np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))])
        spectral_flatness = np.array([np.mean(librosa.feature.spectral_flatness(y=y))])
        rms = np.array([np.mean(librosa.feature.rms(y=y))])
        zcr = np.array([np.mean(librosa.feature.zero_crossing_rate(y=y))])

        # Combine features
        return np.concatenate((mfccs, chroma, mel, spectral_contrast,
                               spectral_centroid, spectral_bandwidth,
                               spectral_flatness, rms, zcr))
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None


# Read attributes
attributes_df = pd.read_csv(ATTRIBUTE_FILE)

# Feature extraction
features, labels = [], []
for index, row in attributes_df.iterrows():
    file_name = row['filename']
    story_type = row['Story_type']
    file_path = os.path.join(AUDIO_FOLDER, file_name)

    # Extract features
    feature = extract_features(file_path)
    if feature is not None:
        features.append(feature)
        labels.append(story_type)

print('Features extracted.')

# Convert to DataFrame and save
features_df = pd.DataFrame(features)
features_df['label'] = labels
features_df.to_csv(FEATURES_FILE, index=False)
