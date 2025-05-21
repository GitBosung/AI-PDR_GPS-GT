import numpy as np
import os
import logging
from datetime import datetime
import joblib

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, GlobalAveragePooling1D, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Loss

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, Normalizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class EuclideanWeightedLoss(Loss):
    """
    ℓ = Σ (||Δl̃ - Δl||^2 + κ * ||Δψ̃ - Δψ||^2)
    
    - y_true, y_pred shape = (batch, 2)  →  [Δl, Δψ]
    - reduction 기본값은 "auto" (배치 평균)입니다.
    """
    def __init__(self, kappa=2.0,
                 reduction=tf.keras.losses.Reduction.AUTO,
                 name="euclidean_weighted_loss"):
        super().__init__(reduction=reduction, name=name)
        self.kappa = kappa

    def call(self, y_true, y_pred):
        # 1) 각 성분 분리
        delta_l_true   = y_true[..., 0]   # shape = (batch,)
        delta_psi_true = y_true[..., 1]
        delta_l_pred   = y_pred[..., 0]
        delta_psi_pred = y_pred[..., 1]

        # 2) 제곱 오차
        se_l   = tf.square(delta_l_pred   - delta_l_true)
        se_psi = tf.square(delta_psi_pred - delta_psi_true)

        # 3) 가중합: 논문 식대로 κ 곱해주기
        loss_sample = se_l + self.kappa * se_psi  # shape = (batch,)

        # 4) reduction: 기본적으로 배치 평균으로 반환
        return tf.reduce_mean(loss_sample)


class ModelTrainer:
    """
    LSTM 기반 모델 구축·학습·평가·저장
      - 센서별 스케일링
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
        adam = Adam(
            learning_rate=1e-5,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-6
        )

        self.model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(self.window_size, self.num_features)),
            #BatchNormalization(),
            # Dropout(0.3),

            LSTM(64, return_sequences=True),
            #BatchNormalization(),
            # Dropout(0.3),

            LSTM(32, return_sequences=False),
            # GlobalAveragePooling1D(),
            
            Dense(2)  # speed, heading_change 예측
        ])

        loss_fn = EuclideanWeightedLoss()

        self.model.compile(
            optimizer=adam,
            loss='mse',
            #loss='mse',
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
            X, Y, test_size=0.2, random_state=1217
        )

        X_train = self.scale_sensor_data(X_train, fit=True).astype(np.float32)
        X_test  = self.scale_sensor_data(X_test,  fit=False).astype(np.float32)
        Y_train = Y_train.astype(np.float32)
        Y_test  = Y_test.astype(np.float32)

        self.build_model()

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10),
            #ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
        ]

        history = self.model.fit(
            X_train, Y_train,
            validation_data=(X_test, Y_test),
            batch_size=self.batch_size,
            epochs=self.epochs,
            #callbacks=callbacks,
            verbose=1
        )
        return history, (X_train, Y_train, X_test, Y_test)

    def plot_training_history(self, history):
        plt.figure(figsize=(8, 5))
        epochs = range(1, len(history.history['loss']) + 1)
        plt.plot(epochs, history.history['loss'],     label='Train Loss')
        plt.plot(epochs, history.history['val_loss'], label='Val Loss')
        plt.xlabel('Epochs')
        plt.ylabel('MultiTask MSE Loss')
        plt.title('Training & Validation Loss')
        plt.legend()
        plt.grid()
        plt.show()

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
