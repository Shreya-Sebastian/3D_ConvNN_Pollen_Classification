import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load data from pickle file
model_name = 'ResNet3D_50_epochs'
with open(f'C:/pollen_classification/saved_models/{model_name}/results.pkl', 'rb') as file:
    results = pickle.load(file)

# Label data
epochs = [result[0] for result in results]
train_loss = [result[1] for result in results]
valid_loss = [result[2] for result in results]
train_f1 = [result[3] for result in results]
valid_f1 = [result[4] for result in results]
train_acc = [result[5] for result in results]
valid_acc = [result[6] for result in results]

# Function to smooth data using a moving average window
def smooth_data(data, window_size=6):
    smoothed_data = []  # List to store smoothed data
    for i in range(len(data)):
        # Calculate the range of indices within the moving window
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(data), i + window_size // 2 + 1)
        # Create a window of indices within the specified range
        window = np.linspace(start_idx, end_idx - 1, end_idx - start_idx, dtype=int)
        # Calculate the smoothed value by taking the mean of data points within the window
        smoothed_value = np.mean([data[idx] for idx in window])
        # Append the smoothed value to the list
        smoothed_data.append(smoothed_value)
    return smoothed_data

# Smooth the training and validation data
train_loss = smooth_data(train_loss)
valid_loss = smooth_data(valid_loss)
train_f1 = smooth_data(train_f1)
valid_f1 = smooth_data(valid_f1)
train_acc = smooth_data(train_acc)
valid_acc = smooth_data(valid_acc)

# Plotting
plt.figure(figsize=(12, 6))

# Loss plot
plt.subplot(1, 3, 1)    # To plot each metric in one figure (index 1)
plt.plot(epochs, train_loss, label='Train')
plt.plot(epochs, valid_loss, label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train and Validation Loss')
plt.legend()

# F1-score plot
plt.subplot(1, 3, 2)
plt.plot(epochs, train_f1, label='Train')
plt.plot(epochs, valid_f1, label='Validation')
plt.xlabel('Epoch')
plt.ylabel('F1-score')
plt.title('Train and Validation F1-score')
plt.legend()

# Accuracy plot
plt.subplot(1, 3, 3)
plt.plot(epochs, train_acc, label='Train')
plt.plot(epochs, valid_acc, label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Train and Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.show()