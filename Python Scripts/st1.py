import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    cols = list(df.columns)[1:8]
    df_for_training = df[cols].astype(float)
    scaler = StandardScaler()
    scaler = scaler.fit(df_for_training)
    df_for_training_scaled = scaler.transform(df_for_training)
    return df_for_training_scaled, scaler

def train_model(trainX, trainY):
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
    model.add(LSTM(32, activation='relu', return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(trainY.shape[1]))
    model.compile(optimizer='adam', loss='mse')
    model.fit(trainX, trainY, epochs=50, batch_size=4, validation_split=0.1, verbose=0)
    return model

def predict_WQI(model, trainX, scaler):
    prediction = model.predict(trainX)
    prediction_copies = np.repeat(prediction, trainX.shape[2], axis=-1)
    y_pred_future = scaler.inverse_transform(prediction_copies)[:, 0]
    return y_pred_future[-1]

def get_WQI_bucket(x):
    if x >= 0 and x <= 25:
        return "Excellent"
    elif x >= 26 and x <= 50:
        return "Good"
    elif x >= 51 and x <= 75:
        return "Poor"
    elif x >= 76 and x < 100:
        return "Very Poor"
    else:
        return "Not Suitable for drinking"

def main():
    st.title("Water Quality Index Prediction")

    uploaded_file = st.file_uploader("C:/Users/91938/Downloads/yukthi (2)/yukthi/Basara.csv", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        df_for_training_scaled, scaler = preprocess_data(df)

        n_future = 1
        n_past = 3
        trainX = []
        trainYP = []

        for i in range(n_past, len(df_for_training_scaled) - n_future + 1):
            trainX.append(df_for_training_scaled[i - n_past:i, 0:df_for_training_scaled.shape[1]])
            trainYP.append(df_for_training_scaled[i + n_future - 1:i + n_future, 0])

        trainX, trainYP = np.array(trainX), np.array(trainYP)

        model = train_model(trainX, trainYP)

        WQI = predict_WQI(model, trainX[-1:], scaler)

        WQI_bucket = get_WQI_bucket(WQI)

        st.write(f"Water Quality Index (WQI): {WQI}")
        st.write(f"WQI Bucket: {WQI_bucket}")


if __name__ == "__main__":
    main()
