import os
import json
import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from groq import Groq

# Streamlit page configuration
st.set_page_config(
    page_title="Heart Disease Prediction & LLAMA Chat",
    page_icon="❤️",
    layout="centered"
)

# Link to Chettinad Hospital for patient reference
st.markdown("For more information on heart health, visit [Chettinad Hospital](https://www.chettinadhospital.com/).")


# Load and preprocess dataset
@st.cache_resource
def load_and_preprocess_data():
    data = pd.read_csv('heart.csv')
    X = data.drop('target', axis=1)
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
    X_test_scaled = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data()


# Build and train the CNN model
@st.cache_resource
def build_and_train_model():
    model = Sequential([
        Conv1D(64, 2, activation='relu', input_shape=(X_train.shape[1], 1)),
        MaxPooling1D(2),
        Conv1D(32, 2, activation='relu'),
        MaxPooling1D(2),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

    # Evaluate model accuracy on the test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    return model, test_accuracy


model, test_accuracy = build_and_train_model()

# Display model accuracy
st.write(f"#### Model Test Accuracy: {test_accuracy * 100:.2f}%")

# Input for heart disease prediction
st.title("❤️ Heart Disease Prediction")
st.subheader("Enter Patient Details:")
age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
cp = st.selectbox("Chest Pain Type",
                  ["Typical Angina (0)", "Atypical Angina (1)", "Non-anginal Pain (2)", "Asymptomatic (3)"])
cp = int(cp.split("(")[-1].strip(")"))
trestbps = st.number_input("Resting Blood Pressure", min_value=50, max_value=200, value=120)
chol = st.number_input("Serum Cholesterol", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL (1 = Yes, 0 = No)", [0, 1])
restecg = st.selectbox("Resting ECG", ["Normal (0)", "ST-T wave abnormality (1)", "LV hypertrophy (2)"])
restecg = int(restecg.split("(")[-1].strip(")"))
thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=250, value=150)
exang = st.selectbox("Exercise Induced Angina (0 = No, 1 = Yes)", [0, 1])
oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0)
slope = st.selectbox("Slope of Peak Exercise ST Segment", ["Upsloping (0)", "Flat (1)", "Downsloping (2)"])
slope = int(slope.split("(")[-1].strip(")"))
ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia", ["Normal (1)", "Fixed Defect (2)", "Reversible Defect (3)"])
thal = int(thal.split("(")[-1].strip(")"))

# Prediction
if st.button("Predict"):
    input_data = scaler.transform(
        [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    input_data = input_data.reshape(1, -1, 1)
    prediction = model.predict(input_data)
    result = "At Risk of Heart Disease" if prediction[0][0] > 0.5 else "Low Risk of Heart Disease"
    st.write(f"**Prediction:** {result}")

# LLAMA 3.1 Chatbot in Sidebar
st.sidebar.title("🦙 LLAMA 3.1 Chatbot")
# config_data = json.load(open("config(2).json"))
# GROQ_API_KEY = config_data["GROQ_API_KEY"]
# os.environ["GROQ_API_KEY"] = GROQ_API_KEY
# client = Groq()
client = Groq(
    api_key=st.secrets["GROQ_API_KEY"]
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.sidebar.chat_message(message["role"]):
        st.sidebar.markdown(message["content"])

user_prompt = st.sidebar.chat_input("Ask LLAMA...")
if user_prompt:
    st.sidebar.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})
    response = client.chat.completions.create(model="llama-3.1-8b-instant", messages=st.session_state.chat_history)
    assistant_response = response.choices[0].message.content
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
    st.sidebar.chat_message("assistant").markdown(assistant_response)