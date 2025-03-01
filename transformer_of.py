import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load data
def load_data(file_path):
    # Read CSV file
    df = pd.read_csv(file_path)
    # Convert datetime to proper format if needed
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
    return df

# Create sequences from time series data
def create_sequences(data, target_column, sequence_length=8, prediction_window=4):
    """
    Create sequences for time series prediction
    sequence_length: number of time steps to look back (8 rows = 120 minutes)
    prediction_window: number of steps to predict ahead (4 rows = 60 minutes)
    """
    X, y = [], []
    for i in range(len(data) - sequence_length - prediction_window):
        # Input sequence (8 time steps)
        X.append(data[i:i+sequence_length])
        # Target is the label after prediction_window (binary classification)
        y.append(data.iloc[i+sequence_length+prediction_window][target_column])
    return np.array(X), np.array(y)

# Process data function
def process_data(df, sequence_length=8, prediction_window=4, test_size=0.4):
    # Select relevant features (dropping datetime and label columns)
    feature_columns = df.columns.difference(['datetime', 'label'])
    df_features = df[feature_columns]
    
    # Scale the features
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(df_features)
    scaled_df = pd.DataFrame(scaled_features, columns=feature_columns)
    
    # Add the label column back
    scaled_df['label'] = df['label']
    
    # Create sequences
    X, y = create_sequences(scaled_df, 'label', sequence_length, prediction_window)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    
    return X_train, X_test, y_train, y_test, scaler, feature_columns

# Build transformer model
def build_transformer_model(input_shape, head_size=256, num_heads=4, ff_dim=4, num_transformer_blocks=4, mlp_units=[128], dropout=0.2):
    inputs = layers.Input(shape=input_shape)
    
    # Transformer blocks
    x = inputs
    for _ in range(num_transformer_blocks):
        # Multi-head self-attention
        attention_output = layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(x, x)
        x = layers.LayerNormalization(epsilon=1e-6)(attention_output + x)
        
        # Feed-forward network
        ffn = layers.Dense(ff_dim, activation="relu")(x)
        ffn = layers.Dense(input_shape[-1])(ffn)
        ffn = layers.Dropout(dropout)(ffn)
        x = layers.LayerNormalization(epsilon=1e-6)(x + ffn)
    
    # Global average pooling
    x = layers.GlobalAveragePooling1D()(x)
    
    # MLP for classification
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(dropout)(x)
    
    # Output layer (binary classification)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    
    return models.Model(inputs, outputs)

# Training function with history plotting
def train_and_evaluate(model, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate model
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # Print evaluation metrics
    print("\nTest Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['-1', '+1']))
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return model, history, y_pred

# Save model
def save_model(model, model_path="time_series_transformer_model"):
    model.save(model_path)
    print(f"Model saved to {model_path}")

# Prediction function
def predict_next_outcome(model, latest_data, scaler, feature_columns, sequence_length=8):
    """
    Predict the next outcome (+1 or -1) using the latest data
    latest_data: DataFrame with the latest data (at least sequence_length rows)
    """
    # Prepare the latest data (must have at least sequence_length rows)
    if len(latest_data) < sequence_length:
        raise ValueError(f"Latest data must have at least {sequence_length} rows")
    
    # Get the last sequence_length rows
    latest_sequence = latest_data.iloc[-sequence_length:].copy()
    
    # Scale the features
    features = latest_sequence[feature_columns]
    scaled_features = scaler.transform(features)
    
    # Reshape to match model input shape (1, sequence_length, n_features)
    X = scaled_features.reshape(1, sequence_length, len(feature_columns))
    
    # Predict
    prediction_prob = model.predict(X)[0][0]
    predicted_class = 1 if prediction_prob > 0.5 else -1
    
    return predicted_class, prediction_prob

# Main execution
def main():
    # File path
    file_path = "timeseries_data.csv"  # Update with your file path
    
    # Load data
    df = load_data(file_path)
    
    # Process data
    X_train, X_test, y_train, y_test, scaler, feature_columns = process_data(
        df, 
        sequence_length=8,
        prediction_window=4,  # 4 rows = 60 minutes for 15-minute data
        test_size=0.4
    )
    
    # Model parameters
    input_shape = X_train.shape[1:]  # (sequence_length, n_features)
    
    # Build model
    model = build_transformer_model(
        input_shape=input_shape,
        head_size=256,
        num_heads=4,
        ff_dim=512,
        num_transformer_blocks=4,
        mlp_units=[128, 64],
        dropout=0.3
    )
    
    # Model summary
    model.summary()
    
    # Train and evaluate
    model, history, y_pred = train_and_evaluate(
        model, 
        X_train, y_train, 
        X_test, y_test,
        epochs=100,
        batch_size=32
    )
    
    # Save model
    save_model(model, "time_series_transformer_model")
    
    # Example prediction with the latest data
    latest_data = df.iloc[-8:].copy()
    predicted_class, probability = predict_next_outcome(
        model, latest_data, scaler, feature_columns
    )
    
    print(f"\nPredicted next outcome: {predicted_class} with probability: {probability:.4f}")
    
if __name__ == "__main__":
    main()

# Example usage after model is saved:
"""
# Load saved model
loaded_model = tf.keras.models.load_model("time_series_transformer_model")

# Load the latest data (8 most recent time steps)
latest_data = pd.read_csv("latest_data.csv")

# Make prediction
predicted_class, probability = predict_next_outcome(
    loaded_model, latest_data, scaler, feature_columns
)
print(f"Predicted next outcome: {predicted_class} with probability: {probability:.4f}")
"""
