import numpy as np
import os
from datetime import datetime
import joblib

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization, Dropout

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class ModelTrainer:
    """
    LSTM 기반 모델 구축·학습·평가·저장
      - 센서별 스케일링 (MinMaxScaler)
      - GT(Y) 속도·헤딩 변화 각각 MinMaxScaler(feature_range=(-1,1))
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

        # 센서 데이터(8채널)에 대한 MinMaxScaler를 담을 dict
        self.sensor_scalers = {}

        # GT 속도에 대한 MinMaxScaler(feature_range=(-1,1))
        self.y_speed_scaler = None
        # GT 헤딩 변화량에 대한 MinMaxScaler(feature_range=(-1,1))
        self.y_hc_scaler    = None

        self.model = None

    def build_model(self):
        adam = Adam(
            learning_rate=1e-3,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-6
        )

        self.model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(self.window_size, self.num_features)),
            Dropout(0.3),
            
            LSTM(64, return_sequences=False),
            Dropout(0.3),
            

            Dense(2)  # 출력: [scaled_speed, scaled_heading_change]
        ])

        self.model.compile(
            optimizer=adam,
            loss='mse',
        )
        return self.model

    def scale_sensor_data(self, X, fit: bool = True):
        """
        X: shape = (n_samples, window_size, num_features)
        - axes 딕셔너리를 기준으로 0~7 인덱스를 MinMaxScaler(feature_range=(-1,1))
        - 나머지 피처(feat > 8)는 그대로 통과
        """
        n_samples, win, feat = X.shape
        X_scaled = np.zeros_like(X, dtype=np.float32)

        axes = {
            'Accelerometer x': 0, 'Accelerometer y': 1, 'Accelerometer z': 2,
            'Gyroscope x':     3, 'Gyroscope y':     4, 'Gyroscope z':     5,
            'Acc_Norm':        6, 'Gyro_Norm':       7
        }

        for name, idx in axes.items():
            col = X[:, :, idx].reshape(-1, 1)  # shape = (n_samples*win, 1)
            if fit:
                scaler = MinMaxScaler(feature_range=(-1, 1))
                scaled = scaler.fit_transform(col)
                self.sensor_scalers[name] = scaler
            else:
                scaled = self.sensor_scalers[name].transform(col)
            X_scaled[:, :, idx] = scaled.reshape(n_samples, win)

        # 만약 feat > len(axes) (즉 8) 이면, 그 이후 인덱스들은 그대로 복사
        if feat > len(axes):
            X_scaled[:, :, len(axes):] = X[:, :, len(axes):]

        return X_scaled

    def train_model(self, X, Y):
        """
        1) train/test split
        2) X_train/X_test → sensor_scaling
        3) Y_train/Y_test → GT 속도·헤딩 각각 MinMaxScaler 적용
        4) 모델 빌드 후 train
        5) history와 (X_train, Y_train_scaled, X_test, Y_test_scaled) 반환
        """

        # -------------------------------
        # 1) Train/Test 분리
        # -------------------------------
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=1217
        )

        # -------------------------------
        # 2) X 스케일링 (fit=X_train, transform both)
        # -------------------------------
        X_train = self.scale_sensor_data(X_train, fit=True).astype(np.float32)
        X_test  = self.scale_sensor_data(X_test,  fit=False).astype(np.float32)

        # -------------------------------
        # 3) Y (GT) 속도·헤딩 각각 MinMaxScaler(feature_range=(-1,1))
        # -------------------------------
        # 속도 스케일러
        self.y_speed_scaler = MinMaxScaler(feature_range=(-1, 1))
        speed_train = Y_train[:, 0].reshape(-1, 1)  # shape = (n_train, 1)
        speed_test  = Y_test[:, 0].reshape(-1, 1)
        scaled_speed_train = self.y_speed_scaler.fit_transform(speed_train)
        scaled_speed_test  = self.y_speed_scaler.transform(speed_test)

        # 헤딩 변화량 스케일러
        self.y_hc_scaler = MinMaxScaler(feature_range=(-1, 1))
        hc_train = Y_train[:, 1].reshape(-1, 1)  # shape = (n_train, 1)
        hc_test  = Y_test[:, 1].reshape(-1, 1)
        scaled_hc_train = self.y_hc_scaler.fit_transform(hc_train)
        scaled_hc_test  = self.y_hc_scaler.transform(hc_test)

        # 두 컬럼을 합쳐서 Y_train_scaled, Y_test_scaled 생성
        Y_train_scaled = np.hstack([scaled_speed_train, scaled_hc_train]).astype(np.float32)
        Y_test_scaled  = np.hstack([scaled_speed_test,  scaled_hc_test ]).astype(np.float32)

        # -------------------------------
        # 4) 모델 빌드
        # -------------------------------
        self.build_model()

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10),
        ]

        history = self.model.fit(
            X_train, Y_train_scaled,
            validation_data=(X_test, Y_test_scaled),
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=callbacks,
            verbose=1
        )

        return history, (X_train, Y_train_scaled, X_test, Y_test_scaled)

    def plot_training_history(self, history):
        """
        Train/Validation Loss 곡선 그리기
        """
        plt.figure(figsize=(8, 5))
        epochs = range(1, len(history.history['loss']) + 1)
        plt.plot(epochs, history.history['loss'],     label='Train Loss')
        plt.plot(epochs, history.history['val_loss'], label='Val Loss')
        plt.xlabel('Epochs')
        plt.ylabel('MSE Loss')
        plt.title('Training & Validation Loss')
        plt.legend()
        plt.grid()
        plt.show()

    def save_model(self, model_dir='saved_models'):
        """
        모델(.h5) 및 scaler(joblib) 저장
        - self.sensor_scalers (dict)
        - self.y_speed_scaler
        - self.y_hc_scaler
        """
        os.makedirs(model_dir, exist_ok=True)
        ts    = datetime.now().strftime('%Y%m%d_%H%M%S')
        mpath = os.path.join(model_dir, f'model_{ts}.h5')
        spath = os.path.join(model_dir, f'scalers_{ts}.joblib')

        # 1) Keras 모델 저장
        self.model.save(mpath)

        # 2) Scaler 묶음 저장 (sensor_scalers, y_speed_scaler, y_hc_scaler)
        joblib.dump({
            'sensor': self.sensor_scalers,
            'y_speed': self.y_speed_scaler,
            'y_hc':    self.y_hc_scaler
        }, spath)

        return mpath

    def load_model(self, model_path):
        """
        1) Keras 모델 로드
        2) 참조하는 스케일러(joblib) 로드 (sensor_scalers, y_speed_scaler, y_hc_scaler 복원)
        """
        self.model = load_model(model_path, compile=False)
        spath = model_path.replace('model_', 'scalers_').replace('.h5', '.joblib')

        if os.path.exists(spath):
            data = joblib.load(spath)
            self.sensor_scalers = data['sensor']
            self.y_speed_scaler = data['y_speed']
            self.y_hc_scaler    = data['y_hc']

        return self.model 