import librosa
import numpy as np

def preprocess_audio(audio_path: str, sr=16000):
    """
    Load audio, convert to mono 16kHz, compute log-mel spectrogram.
    Returns numpy array of shape (n_mels, frames)
    """
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    # trim silence
    y, _ = librosa.effects.trim(y)
    # log-mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y, sr=sr, n_mels=80)
    log_mel_spec = librosa.power_to_db(mel_spec)
    return log_mel_spec
