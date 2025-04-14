import os
import numpy as np
import librosa

def decode_emotion_code(code):
    return {
        'W': 'sadness',
        'L': 'boredom',
        'E': 'disgust',
        'A': 'fear',
        'F': 'anger',
        'T': 'happiness',
        'N': 'neutral'
    }.get(code.upper(), 'unknown')

def extract_mfcc(y, sr, n_mfcc=13):
    return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

def process_audio_file(filepath, features_dir="features"):
    filename = os.path.basename(filepath)
    y, sr = librosa.load(filepath, sr=None)
    mfcc = extract_mfcc(y, sr)
    
    mfcc_filename = filename.replace(".wav", "_mfcc.npy")
    np.save(os.path.join(features_dir, mfcc_filename), mfcc)
    
    emotion = decode_emotion_code(filename[5])
    
    return {
        "audio_file": filename,
        "emotion": emotion,
        "mfcc_file": mfcc_filename
    }
