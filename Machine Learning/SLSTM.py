import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from Ingestor import Ingestor
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, SimpleRNN, Dense, Dropout, Bidirectional, Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def initWeights(rows, cols):
    return np.random.randn(rows, cols) * 0.01  # Small random initialization

class SLSTM:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize weight matrices and biases
        self.wf_x = initWeights(hidden_size, input_size)
        self.wf_h = initWeights(hidden_size, hidden_size)
        self.wi_x = initWeights(hidden_size, input_size)
        self.wi_h = initWeights(hidden_size, hidden_size)
        self.wo_x = initWeights(hidden_size, input_size)
        self.wo_h = initWeights(hidden_size, hidden_size)
        self.wc_x = initWeights(hidden_size, input_size)
        self.wc_h = initWeights(hidden_size, hidden_size)

        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))

        self.wy = initWeights(output_size, hidden_size)
        self.by = np.zeros((output_size, 1))

        self.reset()

    def reset(self):
        self.hidden_states = {-1: np.zeros((self.hidden_size, 1))}
        self.cell_states = {-1: np.zeros((self.hidden_size, 1))}
        self.forget_gates = {}
        self.input_gates = {}
        self.candidate_gates = {}
        self.output_gates = {}

    def forward(self, X):
        T = X.shape[1]  # Time steps
        self.outputs = {}
        prev_h = np.zeros((self.hidden_size, 1))
        prev_c = np.zeros((self.hidden_size, 1))
        for t in range(T):
            x_t = X[:, t, :].reshape(-1, 1)

            # Ensure dictionary entries exist for t
            if t not in self.hidden_states:
                self.hidden_states[t] = np.zeros((self.hidden_size, 1))
                self.cell_states[t] = np.zeros((self.hidden_size, 1))
            if t > 0:
                prev_h = self.hidden_states[t - 1]
                prev_c = self.cell_states[t - 1]

            # Compute gate activations
            self.forget_gates[t] = sigmoid(np.dot(self.wf_x, x_t) + np.dot(self.wf_h, prev_h) + self.bf)
            self.input_gates[t] = sigmoid(np.dot(self.wi_x, x_t) + np.dot(self.wi_h, prev_h) + self.bi)
            self.candidate_gates[t] = np.tanh(np.dot(self.wc_x, x_t) + np.dot(self.wc_h, prev_h) + self.bc)
            self.output_gates[t] = sigmoid(np.dot(self.wo_x, x_t) + np.dot(self.wo_h, prev_h) + self.bo)

            # Update cell state and hidden state
            self.cell_states[t] = self.forget_gates[t] * prev_c + self.input_gates[t] * self.candidate_gates[t]
            self.hidden_states[t] = self.output_gates[t] * np.tanh(self.cell_states[t])

            # Compute output
            self.outputs[t] = np.dot(self.wy, self.hidden_states[t]) + self.by

        return self.outputs

    def backward(self, X, Y, outputs):
        if len(X.shape) == 2:
            X = X.reshape((X.shape[0], 2, X.shape[1]))
        T = X.shape[1]
        dWy = np.zeros_like(self.wy)
        dBy = np.zeros_like(self.by)

        dH_next = np.zeros((self.hidden_size, 1))
        dC_next = np.zeros((self.hidden_size, 1))

        for t in reversed(range(T)):
            x_t = X[:, t, :].T
            y_t = Y[:, t, :].T
            output = outputs[t]

            if t not in self.hidden_states:
                self.hidden_states[t] = np.zeros((self.hidden_size, 1))

            # Compute output gradients
            dY = output - y_t
            dWy += np.dot(dY, self.hidden_states[t].T)
            dBy += dY

            # Compute hidden state error
            dH = np.dot(self.wy.T, dY) + dH_next
            dO = dH * np.tanh(self.cell_states[t]) * sigmoid_derivative(self.output_gates[t])
            dC = dH * self.output_gates[t] * (1 - np.tanh(self.cell_states[t]) ** 2) + dC_next
            dF = dC * self.cell_states[t - 1] * sigmoid_derivative(self.forget_gates[t])
            dI = dC * self.candidate_gates[t] * sigmoid_derivative(self.input_gates[t])
            dC_bar = dC * self.input_gates[t] * (1 - self.candidate_gates[t] ** 2)

            # Compute weight and bias gradients
            dWf_x = np.dot(dF, x_t.T)
            dWf_h = np.dot(dF, self.hidden_states[t - 1].T)
            dbf = dF

            dWi_x = np.dot(dI, x_t.T)
            dWi_h = np.dot(dI, self.hidden_states[t - 1].T)
            dbi = dI

            dWo_x = np.dot(dO, x_t.T)
            dWo_h = np.dot(dO, self.hidden_states[t - 1].T)
            dbo = dO

            dWc_x = np.dot(dC_bar, x_t.T)
            dWc_h = np.dot(dC_bar, self.hidden_states[t - 1].T)
            dbc = dC_bar

            # Update next state errors
            dH_next = np.dot(self.wf_h.T, dF) + np.dot(self.wi_h.T, dI) + np.dot(self.wo_h.T, dO) + np.dot(self.wc_h.T, dC_bar)
            dC_next = dC * self.forget_gates[t]

        return dWy, dBy  # Add other weight updates as needed
    
    def train(self, X, Y, epochs=100):
        if len(X.shape) == 2:  # If X is 2D (num_samples, num_features), reshape to 3D (num_samples, time_steps, features)
            X = X.reshape((X.shape[0], 2, X.shape[1]))  # Add a time dimension (1 time step)
        for epoch in range(epochs):
            outputs = self.forward(X)
            dWy, dBy = self.backward(X, Y, outputs)

            # Update weights using gradient descent
            self.wy -= self.learning_rate * dWy
            self.by -= self.learning_rate * dBy

            if epoch % 10 == 0:
                loss = np.mean((np.array(list(outputs.values())) - Y) ** 2)
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, X):
        return self.forward(X)
"""
# Define model parameters
input_size = 3
hidden_size = 5
output_size = 1
learning_rate = 0.01

# Instantiate the LSTM model (without num_epochs and learning_rate)
lstm = SLSTM(input_size, hidden_size, output_size, learning_rate)

# Generate dummy data (time series of length T)
T = 10
X_train = np.random.randn(input_size, T)  # Shape: (input_size, T)
Y_train = np.random.randn(output_size, T) # Shape: (output_size, T)

# Run forward and backward passes
lstm.train(X_train, Y_train, 100)

# Generate dummy test data
X_test = np.random.randn(input_size, T)  # New time series data

# Get predictions
predictions = lstm.forward(X_test)

# Print predictions
print("Predicted Outputs:")
print(predictions)"""  

# Compare with actual values
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load and preprocess data
amnistadRelease = 'DataSetExport-Discharge Total.Last-24-Hour-Change-in-Storage@08450800-Instantaneous-TCM-20240622194957.csv'
data = Ingestor(amnistadRelease).data
df = pd.DataFrame(data)
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
# Ensure 'Timestamp' column is set as index
df.set_index('Timestamp', inplace=True)

# Extract time-based features if index is DatetimeIndex
if isinstance(df.index, pd.DatetimeIndex):
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    df['Day'] = df.index.day
    df['Hour'] = df.index.hour
    df['Minute'] = df.index.minute
else:
    raise TypeError("Index is not a DatetimeIndex. Please check the conversion of the 'Timestamp' column.")

# Create a custom time-based feature To convert 
df['CustomTimeFeature'] = (
    df['Year'] * 10000 +         # Year * 10000 (e.g., 2024 -> 20240000)
    df['Month'] * 100 +          # Month * 100 (e.g., 09 -> 900)
    df['Day'] +                  # Day (e.g., 21)
    df['Hour'] / 24 +            # Hour as a fraction of the day (e.g., 15 -> 0.625)
    df['Minute'] / 1440          # Minute as a fraction of the day (e.g., 30 -> 0.0208)
)

# Scale the custom time feature
scaler_time = MinMaxScaler()
scaled_time_features = scaler_time.fit_transform(df[['CustomTimeFeature']])

# Scale numerical values
scaler_values = MinMaxScaler()
scaled_values = scaler_values.fit_transform(df[['Value']])

# Combine the scaled features
combined_features = np.hstack([scaled_time_features, scaled_values])

def predict_with_custom_value(timestamp, value, model, seq_length):
    timestamp = pd.to_datetime(timestamp)

    custom_time_feature = (
        timestamp.year * 10000 +        # Year * 10000 (e.g., 2024 -> 20240000)
        timestamp.month * 100 +         # Month * 100 (e.g., 09 -> 900)
        timestamp.day +                 # Day (e.g., 21)
        timestamp.hour / 24 +           # Hour as a fraction of the day (e.g., 15 -> 0.625)
        timestamp.minute / 1440         # Minute as a fraction of the day (e.g., 30 -> 0.0208)
    )

    scaled_value = scaler_values.transform(np.array([[value]]))

    combined_features = np.array([[custom_time_feature, scaled_value[0][0]]])

    custom_sequence = combined_features.reshape((1, seq_length + 1, 1)) 

    prediction = model.predict(custom_sequence)
    prediction = prediction.reshape(-1, 1)
    actual_prediction = abs(scaler_values.inverse_transform(prediction))
    print(f"Prediction for the next time step after {timestamp} with input value {value}: {actual_prediction[0][0]}")
    return actual_prediction[0][0]

def create_sequences_with_targets(data, seq_length):
    sequences = []
    targets = []
    
    # Loop through the data to create sequences
    for i in range(seq_length, len(data)):
        # Extract the timestamp
        timestamp = data[i][0]

        # The target is the value at the current timestamp
        target = data[i][1]
        
        # Extract the prior values (seq_length total) before the current index
        prior_values = [data[j][1:] for j in range(i - seq_length  + 1, i + 1)]
        
        # Create a sequence where data[0] = timestamp and the rest are prior values
        sequence = [timestamp] + [item for sublist in prior_values for item in sublist]
        
        sequences.append(sequence)
        targets.append(target)
    
    return np.array(sequences), np.array(targets)
    
# Define sequence length and shift = [['scaledTime', 'scaledValue']]
seq_length = 1
shift = 1 

# Create X and Y
X,y = create_sequences_with_targets(combined_features, seq_length)
# Extract the custom time feature (first column) and values (second column)
scaled_custom_time = X[:, 0]
scaled_values_only = X[:, 1:]

# Inverse transform the time feature
original_custom_time = scaler_time.inverse_transform(scaled_custom_time.reshape(-1, 1))

# Inverse transform the values
original_values = scaler_values.inverse_transform(scaled_values_only.reshape(-1, 1))

# Split into train and test sets
train_size = int(len(X) * 0.8)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# Reshape for LSTM, RNN, and Bidirectional RNN
num_features = X_train.size // (X_train.shape[0] * seq_length)
X_train = X_train.reshape((X_train.shape[0], seq_length + 1, 1))
X_test = X_test.reshape((X_test.shape[0],seq_length + 1, 1))

# Instantiate the LSTM model  
input_size = 525
hidden_size = 5
output_size = 1
learning_rate = 0.01
 
lstm = SLSTM(input_size, hidden_size, output_size, learning_rate)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Expected input size:", lstm.input_size)
lstm.train(X_train, y_train, 100)
lstm_predictions = lstm.predict(y_train)
lstm_predictions = lstm_predictions.reshape(-1, 1)
lstm_predictions = scaler_values.inverse_transform(lstm_predictions)

test_timestamps = df.index[train_size + (seq_length):]
plt.figure(figsize=(14, 7))
plt.plot(test_timestamps, lstm_predictions, label='LSTM Predicted', color='blue')