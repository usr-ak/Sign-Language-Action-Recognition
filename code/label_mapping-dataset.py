import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import concurrent.futures

actions = np.array(['no action', 'ready', 'hello', 'good', 'morning', 'whats up', 'how', 'you', 'i', 'fine', 'nice', 'meet', 'ok', 'thank you', 'see you later', 'good bye'])
KEYPOINTS_PATH = r'D:\P8\others\data\MP_DataNF'
sequence_length = 30

# Map labels
label_map = {label: num for num, label in enumerate(actions)}

# Preload all keypoints into memory
keypoints_cache = {}
for action in actions:
    for sequence in os.listdir(os.path.join(KEYPOINTS_PATH, action)):
        for frame_num in range(sequence_length):
            npy_file = os.path.join(KEYPOINTS_PATH, action, sequence, f"{frame_num}.npy")
            if os.path.exists(npy_file):
                keypoints_cache[npy_file] = np.load(npy_file)

# Prepare dataset using parallel processing and cache
sequences, labels = [], []
for action in actions:
    for sequence in os.listdir(os.path.join(KEYPOINTS_PATH, action)):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(lambda f: keypoints_cache.get(f), os.path.join(KEYPOINTS_PATH, action, sequence, f"{frame_num}.npy")) for frame_num in range(sequence_length)]
            window = [future.result() for future in futures if future.result() is not None]
        if len(window) == sequence_length:
            sequences.append(window)
            labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10/110, stratify=y, random_state=42)
np.savez('dataset_15na.npz', X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)