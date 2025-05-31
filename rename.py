# import os
# import yaml

# def load_config(path="config.yaml"):
#     with open(path, "r") as f:
#         return yaml.safe_load(f)

# def rename_dataset_split(split_path, split_name, class_names):
#     for emotion in class_names:
#         emotion_dir = os.path.join(split_path, emotion)
#         if not os.path.isdir(emotion_dir):
#             continue

#         wav_files = sorted([
#             f for f in os.listdir(emotion_dir)
#             if f.lower().endswith(".wav")
#         ])

#         for idx, fname in enumerate(wav_files):
#             new_name = f"{split_name}_{emotion}_{idx:03d}.wav"
#             old_path = os.path.join(emotion_dir, fname)
#             new_path = os.path.join(emotion_dir, new_name)

#             if old_path != new_path:
#                 os.rename(old_path, new_path)
#                 print(f"Renamed {fname} --> {new_name}")

# def main():
#     cfg = load_config()

#     class_names = cfg["classes"]
#     split_paths = {
#         "train": cfg["data"]["aud_train_dir"],
#         "val": cfg["data"]["aud_val_dir"],
#         "test": cfg["data"]["aud_test_dir"]
#     }

#     for split_name, split_path in split_paths.items():
#         print(f"\n--- Processing split: {split_name} ---")
#         rename_dataset_split(split_path, split_name, class_names)

# if __name__ == "__main__":
#     main()
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

# for audio playback
try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False

# Config
EXAMPLES_ROOT = "features"  # base folder of .npy files by class
AUDIO_ROOT = "dataset/audio/train"  # to optionally match .wav
MEL_BINS = 39

def inspect_file(npy_path, show_wav=False):
    print(f"\nInspecting file: {npy_path}")
    mel_np = np.load(npy_path)

    # 1. Shape check
    print(f"Shape: {mel_np.shape}  (Expected: {MEL_BINS}, T)")

    # 2. Value range stats
    print(f"Min: {mel_np.min():.2f}, Max: {mel_np.max():.2f}")
    print(f"Mean: {mel_np.mean():.2f}, Std: {mel_np.std():.2f}")

    # 3. Visualize spectrogram
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_np, origin="lower", aspect="auto", cmap="magma")
    plt.colorbar(label="dB")
    plt.title(f"Log-Mel Spectrogram\n{os.path.basename(npy_path)}")
    plt.xlabel("Time frames")
    plt.ylabel("Mel bands")
    plt.tight_layout()
    plt.show()

    # 4. play original audio
    if show_wav and TORCHAUDIO_AVAILABLE:
        # try to locate corresponding .wav file
        emotion = os.path.basename(os.path.dirname(npy_path))
        base = os.path.splitext(os.path.basename(npy_path))[0].split("_")[0]
        wav_candidates = [
            f for f in os.listdir(os.path.join(AUDIO_ROOT, emotion))
            if base in f and f.endswith(".wav")
        ]
        if wav_candidates:
            wav_path = os.path.join(AUDIO_ROOT, emotion, wav_candidates[0])
            print(f"Playing original audio: {wav_path}")
            waveform, sr = torchaudio.load(wav_path)
            torchaudio.utils.sox_effects.init_sox_effects()
            torchaudio.utils.sox_effects.shutdown_sox_effects()
            torchaudio.functional.play_audio(waveform, sample_rate=sr)
        else:
            print("Original .wav not found.")
    elif show_wav:
        print("torchaudio not available for playback.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Path to the .npy file to inspect")
    parser.add_argument("--play", action="store_true", help="Try to play corresponding .wav file")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print("File does not exist.")
    else:
        inspect_file(args.file, show_wav=args.play)
