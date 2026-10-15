import sys
import os

# Ensure python looks into the newly structured src directory
sys.path.append(os.path.abspath("src"))

from data_prep import load_and_preprocess_data
# Placeholder fallback path for DTMB 5415 local testing or mock dataset generation
mat_path = "/content/drive/MyDrive/DTMB5415.mat" if os.path.exists("/content/drive/MyDrive/DTMB5415.mat") else "dummy_path.mat"

print("Executing organized structured GitHub repository pipeline...")
X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_data(mat_path)
