import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import pickle

# Dummy training data
X = np.random.rand(1000,9)
y = ((X[:,0]+X[:,4]+X[:,6]+X[:,7])/4 > 0.5).astype(int)

model = GradientBoostingClassifier(n_estimators=100)
model.fit(X,y)

with open("fusion_model.pkl","wb") as f:
    pickle.dump(model,f)

print("Fusion model trained and saved to fusion_model.pkl")
