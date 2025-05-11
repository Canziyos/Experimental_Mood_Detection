import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Visualize waveform and MFCC for a sample file
def visualize_sample(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    plt.figure(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr)
    plt.title("Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()

    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis='time')
    plt.colorbar()
    plt.title("MFCC (13 Coefficients)")
    plt.tight_layout()
    plt.show()

# Visualize emotion distribution
def visualize_distribution(metadata_path="metadata.csv"):
    df = pd.read_csv(metadata_path)
    counts = df["emotion"].value_counts()
    plt.figure(figsize=(8, 4))
    counts.plot(kind='bar', color='skyblue')
    plt.title("Number of Samples per Emotion")
    plt.xlabel("Emotion")
    plt.ylabel("Count")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()
