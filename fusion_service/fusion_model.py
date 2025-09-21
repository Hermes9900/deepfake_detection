import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import pickle
import os

# For demo: train dummy model if not exists
MODEL_PATH = "fusion_model.pkl"
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH,"rb") as f:
        model = pickle.load(f)
else:
    # Dummy model: random training for demo
    X = np.random.rand(100,9)
    y = (X[:,0]+X[:,4]+X[:,6]+X[:,7])/4 > 0.5
    model = GradientBoostingClassifier()
    model.fit(X,y)
    with open(MODEL_PATH,"wb") as f:
        pickle.dump(model,f)

def predict_fusion(features):
    """
    Input: 2D numpy array of shape (1,9)
    Output: fake_probability, reason_codes[]
    """
    prob = model.predict_proba(features)[:,1][0]
    reason_codes = []
    if features[0,4] > 0.7:
        reason_codes.append("image_high_conf")
    if features[0,0] > 0.8:
        reason_codes.append("text_high_conf")
    if features[0,6] > 0.7:
        reason_codes.append("audio_high_conf")
    return prob, reason_codes
