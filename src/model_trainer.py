# src/model_trainer.py

import numpy as np
import os
import logging
from datetime import datetime
import joblib

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, BatchNormalization, Dropout, Dense
from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    LSTM 기반 모델 구축·학습·평가·저장
      - 센서별 스케일링
      - 레이블(Y)도 MinMaxScaler(-1,1)로 스케일링
      - 슬라이딩 윈도우 입력
    """
    def __init__(self,
                 window_size,
                 num_features,
                 epochs=100,
                 batch_size=128):
        self.window_size    = window_size
        self.num_features   = num_features
        self.epochs         = epochs
        self.batch_size     = batch_size

        self.sensor_scalers = {}
        self.model          = None

    def build_model(self):
        adam = Adam(learning_rate=3e-4,
                    beta_1=0.9,
                    beta_2=0.999,
                    epsilon=1e-7)

        self.model = Sequential([
            LSTM(128, return_sequences=True,
                 input_shape=(self.window_size, self.num_features)),

            LSTM(64, return_sequences=True),

            LSTM(32, return_sequences=False),
            
            Dense(2)   # [speed, heading_change]
        ])

        self.model.compile(
            optimizer=adam,
            loss='mse',
            metrics=['mae']
        )
        return self.model

    def scale_sensor_data(self, X, fit: bool = True):
        n_samples, win, feat = X.shape
        X_scaled = np.zeros_like(X)
        axes = {
            'Accelerometer x': 0, 'Accelerometer y': 1, 'Accelerometer z': 2,
            'Gyroscope x':     3, 'Gyroscope y':     4, 'Gyroscope z':     5,
            'Acc_Norm':        6, 'Gyro_Norm':       7
        }
        for name, idx in axes.items():
            col = X[:, :, idx].reshape(-1, 1)
            if fit:
                scaler = MinMaxScaler(feature_range=(-1, 1))
                scaled = scaler.fit_transform(col)
                self.sensor_scalers[name] = scaler
            else:
                scaled = self.sensor_scalers[name].transform(col)
            X_scaled[:, :, idx] = scaled.reshape(n_samples, win)

        if feat > len(axes):
            X_scaled[:, :, len(axes):] = X[:, :, len(axes):]
        return X_scaled

    def train_model(self, X, Y):
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=1217)

        X_train = self.scale_sensor_data(X_train, fit=True).astype(np.float32)
        X_test  = self.scale_sensor_data(X_test,  fit=False).astype(np.float32)
        Y_train = Y_train.astype(np.float32)
        Y_test  = Y_test.astype(np.float32)

        self.build_model()
        history = self.model.fit(
            X_train, Y_train,
            validation_data=(X_test, Y_test),
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=1
        )
        return history, (X_train, Y_train, X_test, Y_test)

    def plot_training_history(self, history):
        plt.figure(figsize=(8, 5))
        epochs = range(1, len(history.history['loss']) + 1)
        plt.plot(epochs, history.history['loss'],     label='Train Loss')
        plt.plot(epochs, history.history['val_loss'], label='Val Loss')
        plt.xlabel('Epochs'); plt.ylabel('MSE Loss')
        plt.title('Training & Validation Loss')
        plt.legend(); plt.grid(); plt.show()

    def save_model(self, model_dir='saved_models'):
        os.makedirs(model_dir, exist_ok=True)
        ts    = datetime.now().strftime('%Y%m%d_%H%M%S')
        mpath = os.path.join(model_dir, f'model_{ts}.h5')
        spath = os.path.join(model_dir, f'scalers_{ts}.joblib')

        self.model.save(mpath)
        joblib.dump({
            'sensor': self.sensor_scalers,
        }, spath)
        return mpath

    def load_model(self, model_path):
        self.model = load_model(model_path, compile=False)
        spath = model_path.replace('model_', 'scalers_').replace('.h5', '.joblib')
        if os.path.exists(spath):
            data = joblib.load(spath)
            self.sensor_scalers = data['sensor']
        return self.model
