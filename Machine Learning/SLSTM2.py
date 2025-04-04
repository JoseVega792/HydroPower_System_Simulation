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

class SLSTM2(tf.keras.Model):
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        super(SLSTM2, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize weight matrices and biases as layers
        self.wf_x = self.add_weight(shape=(hidden_size, input_size), initializer='random_normal', trainable=True)
        self.wf_h = self.add_weight(shape=(hidden_size, hidden_size), initializer='random_normal', trainable=True)
        self.wi_x = self.add_weight(shape=(hidden_size, input_size), initializer='random_normal', trainable=True)
        self.wi_h = self.add_weight(shape=(hidden_size, hidden_size), initializer='random_normal', trainable=True)
        self.wo_x = self.add_weight(shape=(hidden_size, input_size), initializer='random_normal', trainable=True)
        self.wo_h = self.add_weight(shape=(hidden_size, hidden_size), initializer='random_normal', trainable=True)
        self.wc_x = self.add_weight(shape=(hidden_size, input_size), initializer='random_normal', trainable=True)
        self.wc_h = self.add_weight(shape=(hidden_size, hidden_size), initializer='random_normal', trainable=True)

        self.bf = self.add_weight(shape=(hidden_size, 1), initializer='zeros', trainable=True)
        self.bi = self.add_weight(shape=(hidden_size, 1), initializer='zeros', trainable=True)
        self.bo = self.add_weight(shape=(hidden_size, 1), initializer='zeros', trainable=True)
        self.bc = self.add_weight(shape=(hidden_size, 1), initializer='zeros', trainable=True)

        self.wy = self.add_weight(shape=(output_size, hidden_size), initializer='random_normal', trainable=True)
        self.by = self.add_weight(shape=(output_size, 1), initializer='zeros', trainable=True)

    def call(self, X):
        T = X.shape[1]  # Time steps
        outputs = {}
        
        X = tf.cast(X, dtype=tf.float32)
        # Initialize hidden and cell states
        hidden_states = {-1: tf.zeros((self.hidden_size, 1))}
        cell_states = {-1: tf.zeros((self.hidden_size, 1))}
        
        for t in range(T):
            x_t = tf.reshape(X[:, t], (-1, 2))
            x_t = tf.transpose(x_t)

            prev_h = hidden_states[t - 1]
            prev_c = cell_states[t - 1]

            # Compute gate activations
            forget_gate = tf.sigmoid(tf.matmul(self.wf_x, x_t) + tf.matmul(self.wf_h, prev_h) + self.bf)
            input_gate = tf.sigmoid(tf.matmul(self.wi_x, x_t) + tf.matmul(self.wi_h, prev_h) + self.bi)
            candidate_gate = tf.tanh(tf.matmul(self.wc_x, x_t) + tf.matmul(self.wc_h, prev_h) + self.bc)
            output_gate = tf.sigmoid(tf.matmul(self.wo_x, x_t) + tf.matmul(self.wo_h, prev_h) + self.bo)

            # Update cell state and hidden state
            cell_state = forget_gate * prev_c + input_gate * candidate_gate
            hidden_state = output_gate * tf.tanh(cell_state)

            # Compute output
            outputs[t] = tf.matmul(self.wy, hidden_state) + self.by

            hidden_states[t] = hidden_state
            cell_states[t] = cell_state

        return outputs

    def train_step(self, X, Y):
        with tf.GradientTape() as tape:
            outputs = self.call(X)
            loss = tf.reduce_mean(tf.square(outputs[0] - Y))  # Mean squared error for simplicity

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return loss

    def fit(self, X, Y, epochs=100, batch_size=32):
        self.optimizer = Adam(learning_rate=self.learning_rate)
        for epoch in range(epochs):
            for i in range(0, X.shape[0], batch_size):
                batch_X = X[i:i+batch_size]
                batch_Y = Y[i:i+batch_size]

                loss = self.train_step(batch_X, batch_Y)

            print(f"Epoch {epoch+1}, Loss: {loss.numpy()}")

    def predict(self, X):
        return self.call(X)

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
# Ensure proper reshaping before feeding data into the model
X_train = X_train.reshape((X_train.shape[0], seq_length, 2))
X_test = X_test.reshape((X_test.shape[0], seq_length, 2))

# Check the reshaped data
print("X_train shape after reshaping:", X_train.shape)
print("X_test shape after reshaping:", X_test.shape)

# Instantiate and fit the model
slstm = SLSTM2(input_size=2, hidden_size=50, output_size=1, learning_rate=0.001)
slstm.fit(X_train, y_train, epochs=10, batch_size=32)

# Predict and inverse transform the values
slstm_predictions = slstm.predict(X_test)
slstm_predictions = slstm_predictions[0].numpy()
slstm_predictions = slstm_predictions.reshape(-1,1)
slstm_predictions = scaler_values.inverse_transform(slstm_predictions)

'''
# Regular LSTM
lstm_model = Sequential([
    Input(shape=(seq_length + 1, 1)), 
    LSTM(50, return_sequences=True),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])

optimizer = Adam(learning_rate=0.001)
lstm_model.compile(optimizer=optimizer, loss='mean_squared_error')

lstm_model.fit(X_train, y_train, epochs=100, batch_size=1, validation_data=(X_test, y_test))

lstm_predictions = lstm_model.predict(X_test)
lstm_predictions = lstm_predictions.reshape(-1, 1)
lstm_predictions = scaler_values.inverse_transform(lstm_predictions)'''

# Plot the predictions
y_test = scaler_values.inverse_transform(y_test.reshape(-1,1))
test_timestamps = df.index[train_size + seq_length:]
plt.figure(figsize=(14, 7))
print(y_test)
plt.plot(test_timestamps, y_test, label='Actual', color='black')
plt.plot(test_timestamps, slstm_predictions, label='SLSTM Predicted', color='blue')
#plt.plot(test_timestamps, lstm_predictions, label='LSTM Predicted', color='blue')
plt.legend()
plt.show()
