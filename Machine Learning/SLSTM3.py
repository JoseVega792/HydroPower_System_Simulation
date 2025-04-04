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
import numpy as np
import math  # Ensure to import math
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD, RMSprop

class LSTMCell:
    def __init__(self, input_dim, hidden_dim, learning_rate=0.01, optimizer_type='adam'):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        
        # Initialize weights and biases for forget, input, candidate, and output gates as tf.Variable
        self.W_f = np.random.randn(hidden_dim, hidden_dim + input_dim) * 0.01
        self.b_f = np.zeros((hidden_dim, 1))
        
        self.W_i = np.random.randn(hidden_dim, hidden_dim + input_dim) * 0.01
        self.b_i = np.zeros((hidden_dim, 1))
        
        self.W_c = np.random.randn(hidden_dim, hidden_dim + input_dim) * 0.01
        self.b_c = np.zeros((hidden_dim, 1))
        
        self.W_o = np.random.randn(hidden_dim, hidden_dim + input_dim) * 0.01
        self.b_o = np.zeros((hidden_dim, 1))

        # Custom optimizer: SGD or Adam
        if optimizer_type == 'sgd':
            self.optimizer = self.sgd
        elif optimizer_type == 'adam':
            self.optimizer = self.adam
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")

    def get_optimizer(self, optimizer_type):
        optimizers = {
            'adam': Adam(learning_rate=self.learning_rate),
            'sgd': SGD(learning_rate=self.learning_rate),
            'rmsprop': RMSprop(learning_rate=self.learning_rate)
        }
        return optimizers.get(optimizer_type.lower(), Adam(learning_rate=self.learning_rate))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def mse_loss(self, y_pred, y_true):
        # Ensure both arrays are of the same shape
        assert y_pred.shape == y_true.shape, f"Shape mismatch: y_pred {y_pred.shape} vs y_true {y_true.shape}"
        
        # Compute MSE loss using numpy
        return np.mean((y_pred - y_true) ** 2)


    def mse_loss_derivative(self, y_pred, y_true):
        # Compute the MSE loss derivative over the entire sequence (not just the last timestep)
        # For each time step, compute the gradient and sum it
        return 2 * np.sum(y_pred - y_true, axis=0) / y_true.size

    def forward(self, x_t, h_prev, C_prev):
        """Forward pass for a single time step."""
        assert h_prev.shape[0] == self.hidden_dim, f"Expected h_prev shape ({self.hidden_dim}, batch_size), but got {h_prev.shape}"
        assert x_t.shape[0] == self.input_dim, f"Expected x_t shape ({self.input_dim}, batch_size), but got {x_t.shape}"

        # Ensure dimensions match before concatenation
        if h_prev.shape[1] != x_t.shape[1]:
            raise ValueError(f"Dimension mismatch: h_prev has {h_prev.shape[1]} columns, but x_t has {x_t.shape[1]} columns")

        concat = np.concatenate((h_prev, x_t), axis=0)  # Concatenate previous hidden state and input
        
        f_t = self.sigmoid(np.dot(self.W_f, concat) + self.b_f)  # Forget gate
        i_t = self.sigmoid(np.dot(self.W_i, concat) + self.b_i)  # Input gate
        C_tilde = np.tanh(np.dot(self.W_c, concat) + self.b_c)  # Candidate memory
        o_t = self.sigmoid(np.dot(self.W_o, concat) + self.b_o)  # Output gate

        C_t = f_t * C_prev + i_t * C_tilde  # New cell state
        h_t = o_t * np.tanh(C_t)  # New hidden state
        
        return h_t, C_t, f_t, i_t, C_tilde, o_t, concat
    
    def backward(self, dh, dC, f_t, i_t, C_tilde, o_t, C_t, concat, C_prev):
        """Compute the gradients for the LSTM cell during backpropagation."""
        # Backpropagation through output gate
        do_t = dh * np.tanh(C_t) * o_t * (1 - o_t)
        
        # Backpropagation through cell state
        dC_t = dh * o_t * (1 - np.tanh(C_t) ** 2) + dC
        dC_tilde = dC_t * i_t * (1 - C_tilde ** 2)
        di_t = dC_t * C_tilde * i_t * (1 - i_t)
        df_t = dC_t * C_prev * f_t * (1 - f_t)

        # Gradients w.r.t. the weights and biases
        dW_f = np.dot(df_t, concat.T)
        dW_i = np.dot(di_t, concat.T)
        dW_c = np.dot(dC_tilde, concat.T)
        dW_o = np.dot(do_t, concat.T)
        
        db_f = np.sum(df_t, axis=1, keepdims=True)
        db_i = np.sum(di_t, axis=1, keepdims=True)
        db_c = np.sum(dC_tilde, axis=1, keepdims=True)
        db_o = np.sum(do_t, axis=1, keepdims=True)

        # Compute the gradient w.r.t. the input (dx_t) and previous states (dh_next, dC_next)
        dx_t = np.dot(self.W_f.T, df_t) + np.dot(self.W_i.T, di_t) + np.dot(self.W_c.T, dC_tilde) + np.dot(self.W_o.T, do_t)
        dh_next = dx_t[:self.hidden_dim, :]  # Gradient w.r.t. the previous hidden state
        dC_next = df_t * C_t + di_t * C_tilde  # Gradient w.r.t. the previous cell state
        
        return dh_next, dx_t, dC_next, dW_f, dW_i, dW_c, dW_o, db_f, db_i, db_c, db_o


    def update_weights(self, dW_f, dW_i, dW_c, dW_o, db_f, db_i, db_c, db_o):
        """Update weights using the specified optimizer."""
        # Ensure all gradients are converted to TensorFlow variables
        self.W_f, self.b_f = self.optimizer(self.W_f, self.b_f, dW_f, db_f)
        self.W_i, self.b_i = self.optimizer(self.W_i, self.b_i, dW_i, db_i)
        self.W_c, self.b_c = self.optimizer(self.W_c, self.b_c, dW_c, db_c)
        self.W_o, self.b_o = self.optimizer(self.W_o, self.b_o, dW_o, db_o)
    
    def sgd(self, W, b, dW, db):
        """Simple SGD update rule."""
        W -= self.learning_rate * dW
        b -= self.learning_rate * db
        return W, b
    
    def adam(self, W, b, dW, db, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """Adam optimizer update rule."""
        if not hasattr(self, 'm_W'):
            self.m_W = np.zeros_like(W)
            self.v_W = np.zeros_like(W)
            self.m_b = np.zeros_like(b)
            self.v_b = np.zeros_like(b)
        
        # Update moment estimates
        self.m_W = beta1 * self.m_W + (1 - beta1) * dW
        self.v_W = beta2 * self.v_W + (1 - beta2) * (dW ** 2)
        self.m_b = beta1 * self.m_b + (1 - beta1) * db
        self.v_b = beta2 * self.v_b + (1 - beta2) * (db ** 2)
        
        # Bias correction
        m_W_hat = self.m_W / (1 - beta1)
        v_W_hat = self.v_W / (1 - beta2)
        m_b_hat = self.m_b / (1 - beta1)
        v_b_hat = self.v_b / (1 - beta2)
        
        # Parameter updates
        W -= self.learning_rate * m_W_hat / (np.sqrt(v_W_hat) + epsilon)
        b -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + epsilon)
        
        return W, b

class LSTMLayer:
    def __init__(self, input_dim, output_dim, hidden_dim, seq_len, learning_rate=0.001, optimizer_type='adam'):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.output_dim = output_dim  # Added output_dim

        # Initialize LSTM Cell
        self.lstm_cell = LSTMCell(input_dim, hidden_dim)

        # **Initialize Output Projection Layer**
        self.W_out = np.random.randn(self.hidden_dim, self.output_dim)
        self.b_out = np.zeros((self.output_dim,))
    
    def forward(self, X):
        batch_size = X.shape[1]
        h_t = np.zeros((self.hidden_dim, batch_size))
        C_t = np.zeros((self.hidden_dim, batch_size))

        h_states, caches = [], []
        for t in range(self.seq_len):
            x_t = X[t].reshape(self.input_dim, -1)  # Shape should be (input_dim, batch_size)
            print(f"X[t] shape at time step {t}: {X[t].shape}")
            # For the first time step, C_prev is initialized to zeros
            if t > 0:
                C_prev = C_t  # After processing time step t-1, C_t becomes C_prev for time step t
            else:
                C_prev = np.zeros_like(C_t)  # For the first time step, initialize C_prev as zero
            h_t, C_t, f_t, i_t, C_tilde, o_t, concat = self.lstm_cell.forward(x_t, h_t, C_t)
            h_states.append(h_t)
            caches.append((f_t, i_t, C_tilde, o_t, C_t, concat, C_prev))

        h_states = np.stack(h_states)  # Shape: (seq_len, hidden_dim, batch_size)

        # **Apply the output projection**
        h_states_reshaped = h_states.reshape(-1, self.hidden_dim)  # Shape: (batch_size * seq_len, hidden_dim)
        y_pred = np.dot(h_states_reshaped, self.W_out) + self.b_out  # Shape: (batch_size * seq_len, output_dim)
        y_pred = y_pred.reshape(self.seq_len, batch_size, -1)
        # Reshape y_pred back to (seq_len, batch_size, output_dim)
        y_pred = y_pred.reshape(batch_size, -1)

        return y_pred, caches

    
    def backward(self, dh, caches):
        dW_f, dW_i, dW_c, dW_o = 0, 0, 0, 0
        db_f, db_i, db_c, db_o = 0, 0, 0, 0
        dh_next = np.zeros_like(dh[0])
        dC_next = np.zeros_like(dh[0])
        
        for t in reversed(range(self.seq_len)):
            f_t, i_t, C_tilde, o_t, C_t, concat, C_prev = caches[t]
            dh_next, dx_t, dC_next, dW_ft, dW_it, dW_ct, dW_ot, db_ft, db_it, db_ct, db_ot = self.lstm_cell.backward(
                dh[t] + dh_next, dC_next, f_t, i_t, C_tilde, o_t, C_t, concat, C_prev)
            
            dW_f += dW_ft
            dW_i += dW_it
            dW_c += dW_ct
            dW_o += dW_ot
            db_f += db_ft
            db_i += db_it
            db_c += db_ct
            db_o += db_ot
        
        self.lstm_cell.update_weights(dW_f, dW_i, dW_c, dW_o, db_f, db_i, db_c, db_o)
        return dW_f, dW_i, dW_c, dW_o, db_f, db_i, db_c, db_o

    def train(self, X_train, y_train, epochs=10, batch_size=32):
        num_batches = math.ceil(X_train.shape[1] / batch_size)
        print(f"Total batches: {num_batches}")
        
        for epoch in range(epochs):
            total_loss = 0
            for batch in range(num_batches):
                # Fix batch slicing to avoid oversized batches
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, X_train.shape[1])

                batch_X = X_train[start_idx:end_idx]
                batch_y = y_train[start_idx:end_idx]
                print(batch_y)
                # Debugging: Print batch sizes
                print(f"Batch {batch+1}/{num_batches}, batch_X.shape: {batch_X.shape}, batch_y.shape: {batch_y.shape}")

                # Forward pass
                y_pred, caches = self.forward(batch_X)
                print(y_pred)
                # Ensure shape consistency
                assert y_pred.shape == batch_y.shape, f"Shape mismatch: y_pred {y_pred.shape} vs batch_y {batch_y.shape}"

                # Calculate loss (MSE)
                loss = self.lstm_cell.mse_loss(y_pred, batch_y)
                total_loss += loss
                
                # Backward pass
                dh = self.lstm_cell.mse_loss_derivative(y_pred, batch_y)
                if dh.shape[0] != self.seq_len:
                    dh = np.repeat(dh, self.seq_len, axis=0).reshape(self.seq_len, batch_X.shape[1], self.output_dim)
                self.backward(dh, caches)
                
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/num_batches}")

    def predict(self, X):
        """Inference method to predict values without updating weights."""
        batch_size = X.shape[0]  # Ensure correct batch size extraction
        h_t = np.zeros((self.hidden_dim, batch_size))  # Corrected shape
        C_t = np.zeros((self.hidden_dim, batch_size))  # Corrected shape

        h_states = []
        for t in range(X.shape[1]):  # Iterate over timesteps
            x_t = X[:, t, :].T   
            h_t, C_t, _, _, _, _, _ = self.lstm_cell.forward(x_t, h_t, C_t)
            h_states.append(h_t)

        h_states = np.stack(h_states, axis=1)  # Shape: (batch_size, seq_len, hidden_dim)

        return h_states[-1]  # Shape: (batch_size, output_dim)

    '''
# Create dummy data
sequence_length = 10
batch_size = 10
input_dim = 5
output_dim = 1

X_train = np.random.randn(sequence_length, batch_size, input_dim)
y_train = np.random.randn(sequence_length, batch_size, output_dim)

# Initialize LSTM layer with desired parameters
hidden_dim = 8
seq_len = sequence_length
learning_rate = 0.01
optimizer_type = 'adam'
X_train = X_train.reshape((seq_len, batch_size, input_dim))

lstm_layer = LSTMLayer(input_dim, hidden_dim, seq_len, learning_rate, optimizer_type)

# Train the model
epochs = 10
print(f"X_train shape: {X_train.shape}")
lstm_layer.train(X_train, y_train, epochs=epochs, batch_size=batch_size)

# Use the model for inference on new data
X_new = np.random.randn(sequence_length, batch_size, input_dim)  # New input sequence
predictions = lstm_layer.inference(X_new)

print("Predictions:", predictions)
'''
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

train_size = int(len(X) * 0.8)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# Reshape for LSTM, RNN, and Bidirectional RNN
num_features = X_train.size // (X_train.shape[0] * seq_length)
X_train = X_train.reshape((X_train.shape[0], seq_length + 1, 1))  # Add sequence length dimension
X_test = X_test.reshape((X_test.shape[0], seq_length + 1, 1))

# Define and train LSTM model (Keras model)
lstm_model = Sequential([
    Input(shape=(seq_length + 1, 1)),  # Input shape
    LSTM(50, return_sequences=True),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])

optimizer = Adam(learning_rate=0.001)
lstm_model.compile(optimizer=optimizer, loss='mean_squared_error')

# Fit the model on training data
#lstm_model.fit(X_train, y_train, epochs=100, batch_size=1, validation_data=(X_test, y_test))

#lstm_predictions = lstm_model.predict(X_test)
#lstm_predictions = lstm_predictions.reshape(-1, 1)
#lstm_predictions = scaler_values.inverse_transform(lstm_predictions)

# After training, perform inference with the custom LSTM class
# Create the custom LSTM layer (assuming LSTMLayer is your custom class)
input_dim = 2   # Adjust based on the number of features in X
hidden_dim = 50  # Hidden state dimension
seq_len = seq_length  # Sequence length
output_dim = 1  # Adjust based on y shape
learning_rate = 0.001
optimizer_type = 'sgd'

# Ensure proper input shape (num_samples, seq_len, input_dim)
X_train = X_train.reshape((X_train.shape[0], seq_len, input_dim))
X_test = X_test.reshape((X_test.shape[0], seq_len, input_dim))

# Ensure y_train has the correct shape (num_samples, output_dim)
y_train = y_train.reshape((-1, output_dim))

# Create an instance of the custom LSTM layer
lstm_layer = LSTMLayer(input_dim, output_dim, hidden_dim, seq_len, learning_rate, optimizer_type)

# Train the model
epochs = 100
lstm_layer.train(X_train, y_train, epochs=epochs, batch_size=1)

# Use the trained model for inference
predictions = lstm_layer.predict(X_test)
print(predictions.shape)

# Ensure predictions have the correct shape before inverse transformation
predictions = predictions.reshape(-1, 1)  # Taking the last time step

# Convert back to original scale if using MinMaxScaler
predictions = scaler_values.inverse_transform(predictions)

# Print predictions
print(predictions.shape)
print(predictions)
# Plot the predictions
y_test = scaler_values.inverse_transform(y_test.reshape(-1,1))
test_timestamps = df.index[train_size + seq_length:]
print(y_test.shape)
plt.figure(figsize=(14, 7))
plt.plot(test_timestamps, y_test, label='Actual', color='black')
plt.plot(test_timestamps, predictions, label='SLSTM Predicted', color='blue')
#plt.plot(test_timestamps, lstm_predictions, label='LSTM Predicted', color='blue')
plt.legend()
plt.show()