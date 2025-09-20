import numpy as np

def predict_audio(y, sr):
    """
    Dummy audio detector.
    Input: y = audio waveform, sr = sampling rate
    Returns:
    - audio_score: 0-1 probability of fake
    - suspicious_segments: list of (start_sec, end_sec)
    """
    # Split audio into 2s windows
    window_sec = 2
    step = int(sr * window_sec)
    segments = []
    for i in range(0, len(y), step):
        segment = y[i:i+step]
        # placeholder score: variance check for demo
        score = np.random.uniform(0,1)
        if score > 0.7:
            segments.append((i/sr, min((i+step)/sr, len(y)/sr)))
    
    audio_score = np.mean([np.random.uniform(0,1) for _ in range(5)])
    return audio_score, segments
