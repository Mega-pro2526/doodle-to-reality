import numpy as np
import os
from sklearn.model_selection import train_test_split

# Define paths and classes
data_dir = "data/quickdraw_data"
classes = ['cat', 'house', 'tree']
label_map = {cls: idx for idx, cls in enumerate(classes)}

X = []
y = []

# Load data
for cls in classes:
    file_path = os.path.join(data_dir, f"{cls}.npy")
    data = np.load(file_path)
    
    # Limit samples per class to speed things up (optional)
    data = data[:15000]
    
    X.append(data)
    y.extend([label_map[cls]] * len(data))

# Stack arrays
X = np.concatenate(X)
y = np.array(y)

# Normalize pixel values
X = X / 255.0

# Reshape to (samples, 28, 28, 1)
X = X.reshape(-1, 28, 28, 1)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save as .npz
np.savez("data/doodle_dataset.npz", X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
print("✅ Dataset ready for training!")
