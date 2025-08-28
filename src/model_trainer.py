import numpy as np
import os
from datetime import datetime
import joblib

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM, Dense, MultiHeadAttention, LayerNormalization, Add, Input, GlobalAveragePooling1D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class ModelTrainer:
    """
    - build_model: LSTM + MultiHeadAttention 구조의 모델을 생성합니다.
    - scale_sensor_data: 센서 데이터를 스케일링합니다.
    - train_model: 데이터를 분할하고 스케일링한 후 모델을 학습시킵니다.
    """

    def __init__(self, window_size=50, num_features=None, epochs=100, batch_size=128):
        self.window_size = window_size
        self.num_features = num_features
        self.epochs = epochs
        self.batch_size = batch_size
        self.sensor_scalers = {}
        self.y_speed_scaler = None
        self.y_hc_scaler = None
        self.model = None
    
    
    def build_model(self):
        if self.num_features is None:
            raise ValueError("num_features가 설정되지 않았습니다.") 
        
        inputs = Input(shape=(self.window_size, self.num_features))

        # 1) 첫 LSTM
        x = LSTM(128, return_sequences=True)(inputs)
        x = Dropout(0.2)(x)

        # 2) 두 번째 LSTM
        x = LSTM(128, return_sequences=True)(x)
        x = Dropout(0.2)(x)

        # 3) Attention (Self-Attention)
        x_norm = LayerNormalization()(x)
        attn = MultiHeadAttention(num_heads=4, key_dim=32)(x_norm, x_norm)
        x = Add()([x, attn])   # Residual 연결

        # 4) 출력
        x = GlobalAveragePooling1D()(x)  # 시퀀스를 요약 (Dense 전 Flatten 역할)
        outputs = Dense(2)(x)

        self.model = tf.keras.Model(inputs, outputs, name="LSTM_Attention_Simplified")
        self.model.compile(
            optimizer=Adam(learning_rate=1e-4, clipnorm=1.0),
            loss=tf.keras.losses.Huber(delta=1.0),
            metrics=['mae']
        )
        return self.model
    
    def scale_sensor_data(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        n_s, w, f = X.shape
        flat = X.reshape(-1, f)
        scaled = np.zeros_like(flat, dtype=np.float32)

        if fit:
            self.sensor_scalers = {}

        for idx in range(f):
            col = flat[:, idx:idx+1]
            if fit:
                scaler = MinMaxScaler(feature_range=(-1, 1))
                #scaler = StandardScaler()
                flat_s = scaler.fit_transform(col)
                self.sensor_scalers[idx] = scaler
            else:
                flat_s = self.sensor_scalers[idx].transform(col)
            scaled[:, idx] = flat_s.ravel()

        return scaled.reshape(n_s, w, f)

    def train_model(self, X: np.ndarray, Y: np.ndarray):
        X_tr, X_te, Y_tr, Y_te = train_test_split(X, Y, test_size=0.2, random_state=1217)

        self.num_features = X_tr.shape[2]

        X_tr_s = self.scale_sensor_data(X_tr, fit=True)
        X_te_s = self.scale_sensor_data(X_te, fit=False)

        # self.y_speed_scaler = StandardScaler()
        # self.y_hc_scaler = StandardScaler()
        self.y_speed_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.y_hc_scaler    = MinMaxScaler(feature_range=(-1, 1))
        
        y1 = self.y_speed_scaler.fit_transform(Y_tr[:, :1])
        y2 = self.y_hc_scaler.fit_transform(Y_tr[:, 1:2])
        Y_tr_s = np.hstack([y1, y2]).astype(np.float32)

        y1_te = self.y_speed_scaler.transform(Y_te[:, :1])
        y2_te = self.y_hc_scaler.transform(Y_te[:, 1:2])
        Y_te_s = np.hstack([y1_te, y2_te]).astype(np.float32)

        self.build_model()
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, min_delta=1e-4, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, cooldown=2),
            ModelCheckpoint("best_model.h5", monitor="val_loss", save_best_only=True, verbose=0),
        ]
        
        history = self.model.fit(
            X_tr_s, Y_tr_s,
            validation_data=(X_te_s, Y_te_s),
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=callbacks,
            verbose=1
        )
        return history
    
        # train_model 함수를 아래 내용으로 교체!
    # def train_model(self, X_tr: np.ndarray, Y_tr: np.ndarray, X_te: np.ndarray, Y_te: np.ndarray):
    #     """
    #     미리 분할된 훈련(tr) 및 검증(te) 데이터를 받아 모델을 학습시킵니다.
    #     """
    #     # 자동 num_features 설정
    #     self.num_features = X_tr.shape[2]

    #     # 2) X 스케일 (훈련 데이터 기준으로 fit)
    #     X_tr_s = self.scale_sensor_data(X_tr, fit=True)
    #     X_te_s = self.scale_sensor_data(X_te, fit=False)

    #     # 3) Y 스케일 (훈련 데이터 기준으로 fit)
    #     self.y_speed_scaler = StandardScaler()
    #     self.y_hc_scaler    = StandardScaler()
        
    #     y1_tr = self.y_speed_scaler.fit_transform(Y_tr[:, :1])
    #     y2_tr = self.y_hc_scaler.fit_transform(Y_tr[:, 1:2])
    #     Y_tr_s = np.hstack([y1_tr, y2_tr]).astype(np.float32)

    #     y1_te = self.y_speed_scaler.transform(Y_te[:, :1])
    #     y2_te = self.y_hc_scaler.transform(Y_te[:, 1:2])
    #     Y_te_s = np.hstack([y1_te, y2_te]).astype(np.float32)

    #     # 4) 모델 구성 & 학습
    #     self.build_model()
        
    #     callbacks = [
    #         EarlyStopping(monitor='val_loss', patience=15, min_delta=1e-4, restore_best_weights=True),
    #         ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, cooldown=2),
    #         ModelCheckpoint("best_model.h5", monitor="val_loss", save_best_only=True, verbose=0),
    #     ]
        
    #     history = self.model.fit(
    #         X_tr_s, Y_tr_s,
    #         validation_data=(X_te_s, Y_te_s), # 검증 데이터로 X_te_s, Y_te_s 사용
    #         batch_size=self.batch_size,
    #         epochs=self.epochs,
    #         callbacks=callbacks,
    #         verbose=1
    #     )
    #     return history

    def save_model(self, model_dir='saved_models'):
        os.makedirs(model_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        mpath = os.path.join(model_dir, f'model_LSTM_Attention_ws{self.window_size}_{ts}.h5')
        spath = os.path.join(model_dir, f'scalers_LSTM_Attention_ws{self.window_size}_{ts}.joblib')

        self.model.save(mpath)
        joblib.dump({
            'sensor': self.sensor_scalers,
            'y_speed': self.y_speed_scaler,
            'y_hc': self.y_hc_scaler
        }, spath)

        return mpath

    def load_model(self, model_path):
        self.model = load_model(model_path, compile=False)
        spath = model_path.replace('model_', 'scalers_').replace('.h5', '.joblib')
        if os.path.exists(spath):
            data = joblib.load(spath)
            self.sensor_scalers = data['sensor']
            self.y_speed_scaler = data['y_speed']
            self.y_hc_scaler = data['y_hc']
        return self.model
    
    def plot_training_history(self, history):
        plt.figure(figsize=(8, 5))
        epochs = range(1, len(history.history['loss']) + 1)
        plt.plot(epochs, history.history['loss'], label='Train Loss')
        plt.plot(epochs, history.history['val_loss'], label='Val Loss')
        plt.xlabel('Epochs')
        plt.ylabel('MSE Loss')
        plt.title('Training & Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.show()