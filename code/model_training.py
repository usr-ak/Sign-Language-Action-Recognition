import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard, Callback
from sklearn.metrics import classification_report, accuracy_score


# Load preprocessed data
data = np.load(r"D:\P8\SL proj\code\no face\dataset_15na.npz")
X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']


model = Sequential([
    LSTM(64, return_sequences=True, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])), #No. of frames and features per frame
    LSTM(128, return_sequences=True, activation='relu'),
    LSTM(64, return_sequences=False, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(y_train.shape[1], activation='softmax')  # Output layer adjusted to number of actions
])

# Compile model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Callbacks
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)


class StopTrainingOnStablePerformance(Callback):
    def __init__(self, accuracy_threshold=0.95, loss_threshold=0.15, patience=5, smoothing_window=3):
        super().__init__()
        self.accuracy_threshold = accuracy_threshold
        self.loss_threshold = loss_threshold
        self.patience = patience
        self.smoothing_window = smoothing_window
        self.counter = 0

        # Store validation metrics for smoothing
        self.val_acc_history = []
        self.val_loss_history = []

    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs.get('val_categorical_accuracy')
        val_loss = logs.get('val_loss')
        train_acc = logs.get('categorical_accuracy')
        train_loss = logs.get('loss')

        if val_acc is not None and val_loss is not None:
            # Store the last 'smoothing_window' values for moving average
            self.val_acc_history.append(val_acc)
            self.val_loss_history.append(val_loss)

            if len(self.val_acc_history) > self.smoothing_window:
                self.val_acc_history.pop(0)
                self.val_loss_history.pop(0)

            # Compute moving averages
            avg_val_acc = np.mean(self.val_acc_history)
            avg_val_loss = np.mean(self.val_loss_history)

            # Check if conditions are met
            if avg_val_acc >= self.accuracy_threshold and avg_val_loss <= self.loss_threshold:
                self.counter += 1
                print(f"\nEpoch {epoch+1}: Avg Val Acc {avg_val_acc:.4f}, Avg Val Loss {avg_val_loss:.4f} met the threshold ({self.counter}/{self.patience})")
            else:
                self.counter = 0  # Reset counter if conditions are not met

            # Overfitting Check: Large gap between train and validation performance
            if train_acc - avg_val_acc > 0.10 and avg_val_loss > train_loss:
                print(f"\nWarning: Possible overfitting detected! Train Acc: {train_acc:.4f}, Val Acc: {avg_val_acc:.4f}")

            # Stop training if conditions hold for 'patience' epochs
            if self.counter >= self.patience:
                print("\nStopping training as validation accuracy and loss have remained stable for consecutive epochs.")
                self.model.stop_training = True

callback = StopTrainingOnStablePerformance(accuracy_threshold=0.95, loss_threshold=0.15, patience=5, smoothing_window=3)


# Train model
history = model.fit(
    X_train, y_train, 
    validation_data=(X_test, y_test),  # validation data
    epochs=500, 
    callbacks=[tb_callback, callback]
)


# Evaluate model
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

print("Test Accuracy:", accuracy_score(y_test_labels, y_pred))
print(classification_report(y_test_labels, y_pred))


# Save the model
model.save('model_15na.h5')
model.save('model_15na.keras')