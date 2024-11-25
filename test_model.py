import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Fixture to load and preprocess the dataset
@pytest.fixture
def load_and_preprocess_data():
    data = pd.read_csv('heart.csv')
    X = data.drop('target', axis=1)
    y = data['target']

    # Split and preprocess the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Reshape for Conv1D input
    X_train_scaled = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
    X_test_scaled = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

    return X_train_scaled, X_test_scaled, y_train, y_test

# Fixture to create the model
@pytest.fixture
def create_model():
    model = Sequential([
        Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(13, 1)),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=32, kernel_size=2, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')  # Output layer for binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Helper function to train the model
def train_model(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    return model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                     validation_data=(X_val, y_val), callbacks=[early_stopping])

# Helper function to evaluate the model
def evaluate_model(model, X_test, y_test):
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    return test_loss, test_accuracy

# Test Case 1: Ensure test accuracy meets a threshold
def test_baseline_accuracy(load_and_preprocess_data, create_model):
    X_train, X_test, y_train, y_test = load_and_preprocess_data
    model = create_model

    # Train the model
    history = train_model(model, X_train, y_train, X_test, y_test)

    # Evaluate model accuracy
    _, test_accuracy = evaluate_model(model, X_test, y_test)

    # Assert that the test accuracy is above 70%
    assert test_accuracy > 0.7, f"Test Accuracy is below threshold: {test_accuracy:.4f}"

# Test Case 2: Compare validation accuracy and test accuracy
def test_validation_vs_test_accuracy(load_and_preprocess_data, create_model):
    X_train, X_test, y_train, y_test = load_and_preprocess_data
    model = create_model

    # Train the model
    history = train_model(model, X_train, y_train, X_test, y_test)

    # Get validation accuracy from training history
    val_accuracy = history.history['val_accuracy'][-1]

    # Evaluate model accuracy
    _, test_accuracy = evaluate_model(model, X_test, y_test)

    # Assert that validation and test accuracy are close (within 5%)
    assert abs(val_accuracy - test_accuracy) < 0.05, (
        f"Validation Accuracy ({val_accuracy:.4f}) and Test Accuracy ({test_accuracy:.4f}) differ significantly."
    )

# Test Case 3: Check if the model loss is below a threshold
def test_loss_threshold(load_and_preprocess_data, create_model):
    X_train, X_test, y_train, y_test = load_and_preprocess_data
    model = create_model

    # Train the model
    history = train_model(model, X_train, y_train, X_test, y_test)

    # Get the final loss from the history
    final_loss = history.history['val_loss'][-1]

    # Assert that the loss is below a threshold (e.g., 0.5)
    assert final_loss < 0.5, f"Model loss is above threshold: {final_loss:.4f}"

# Test Case 4: Ensure early stopping prevents unnecessary epochs
def test_early_stopping(load_and_preprocess_data, create_model):
    X_train, X_test, y_train, y_test = load_and_preprocess_data
    model = create_model

    # Train the model
    history = train_model(model, X_train, y_train, X_test, y_test, epochs=100)

    # Ensure that the number of epochs used is less than the maximum (early stopping triggered)
    assert len(history.history['loss']) < 100, f"Training continued for {len(history.history['loss'])} epochs."
