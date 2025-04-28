import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization, Dropout
from tensorflow.keras.losses import Huber  
from tensorflow.keras.optimizers import Adam   
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from datetime import datetime
import pandas as pd
import joblib
import math

class ModelTrainer:
    """
    LSTM 기반 모델 구축, 학습, 평가 및 저장을 담당하는 클래스
      - 센서별 + 축별 스케일링 적용 (MinMaxScaler: -1 ~ 1)
      - 슬라이딩 윈도우 방식 입력 데이터 생성
      - 학습 손실 시각화 및 모델 저장
    """
    def __init__(self, window_size, num_features, epochs=100, batch_size=128):
        self.window_size = window_size
        self.num_features = num_features
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.sensor_scalers = {}  # 축별 스케일러 저장

    def build_model(self):
        # 고정 학습률 Adam
        adam = Adam(learning_rate=3e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-7)

        self.model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(self.window_size, self.num_features)),

            LSTM(64, return_sequences=True),

            LSTM(32, return_sequences=False),

            Dense(2)   # [속도, 헤딩 변화량]
        ])

        self.model.compile(
            optimizer=adam,
            loss=Huber(delta=1.0),
            metrics=['mae']
        )
        return self.model

    def analyze_scaling(self, X, X_scaled):
        sensors = {
            'Accelerometer': (0, 3),
            'Gyroscope': (3, 6),
            'Acc_Norm': (6, 7)
        }
        print("\n=== 스케일링 전/후 통계 정보 ===")
        for sensor_name, (start, end) in sensors.items():
            print(f"\n{sensor_name}:")
            for i in range(start, end):
                original = X[:, :, i].flatten()
                scaled = X_scaled[:, :, i].flatten()
                stats_dict = {
                    'Original': {'Min': np.min(original), 'Max': np.max(original), 'Mean': np.mean(original), 'Std': np.std(original)},
                    'Scaled':   {'Min': np.min(scaled),   'Max': np.max(scaled),   'Mean': np.mean(scaled),   'Std': np.std(scaled)},
                }
                stats = pd.DataFrame.from_dict(stats_dict, orient='index')
                print(stats)

        # 히스토그램 시각화
        total_plots = sum(end - start for _, (start, end) in sensors.items())
        n_cols = 4
        n_rows = math.ceil(total_plots / n_cols)

        plt.figure(figsize=(15, 15))
        idx = 1
        for sensor_name, (start, end) in sensors.items():
            for i in range(start, end):
                plt.subplot(n_rows, n_cols, idx)
                idx += 1
                plt.hist(X[:, :, i].flatten(), bins=50, alpha=0.5, label='Original')
                plt.hist(X_scaled[:, :, i].flatten(), bins=50, alpha=0.5, label='Scaled')
                plt.title(f'{sensor_name} idx {i}')
                plt.legend()
        plt.tight_layout()
        plt.show()

    def scale_sensor_data(self, X, fit: bool = True):
        """
        X: (samples, window_size, features)
        fit=True  -> 축별 scaler.fit_transform
        fit=False -> scaler.transform
        """
        n_samples, win, feat = X.shape
        X_scaled = np.zeros_like(X)
        axis_indices = {
            'Accelerometer x': 0, 'Accelerometer y': 1, 'Accelerometer z': 2,
            'Gyroscope x': 3,     'Gyroscope y': 4,     'Gyroscope z': 5,
            'Acc_Norm': 6,
            'rot6_0': 7, 'rot6_1': 8, 'rot6_2': 9, 'rot6_3': 10, 'rot6_4': 11, 'rot6_5': 12  # 모든 Orientation 피처 추가
        }
        for name, idx in axis_indices.items():
            col = X[:, :, idx].reshape(-1, 1)
            if name in ['rot6_0','rot6_1','rot6_2','rot6_3','rot6_4','rot6_5']:  # Orientation 피처는 스케일링하지 않음
                X_scaled[:, :, idx] = col.reshape(n_samples, win)
                continue
            if fit:
                scaler = MinMaxScaler(feature_range=(-1, 1))
                scaled = scaler.fit_transform(col)
                self.sensor_scalers[name] = scaler
            else:
                scaler = self.sensor_scalers[name]
                scaled = scaler.transform(col)
            X_scaled[:, :, idx] = scaled.reshape(n_samples, win)

        # Orientation 등 나머지 피처는 그대로 복사
        if feat > len(axis_indices):
            X_scaled[:, :, len(axis_indices):] = X[:, :, len(axis_indices):]

        if fit:
            self.analyze_scaling(X, X_scaled)
        return X_scaled

    def train_model(self, X, Y):
        # 1) train/test split
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=1217)
        # 2) fit & transform on train, transform on test
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
        train_loss = history.history['loss']
        val_loss   = history.history['val_loss']
        epochs_rng = range(1, len(train_loss) + 1)
        plt.figure(figsize=(8, 5))
        plt.plot(epochs_rng, train_loss, label='Training Loss')
        plt.plot(epochs_rng, val_loss,   label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training & Validation Loss')
        plt.legend()
        plt.grid()
        plt.show()

    def save_model(self, model_dir='saved_models'):
        os.makedirs(model_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        mpath = os.path.join(model_dir, f'model_{ts}.h5')
        spath = os.path.join(model_dir, f'scalers_{ts}.joblib')
        self.model.save(mpath)
        joblib.dump(self.sensor_scalers, spath)
        return mpath

    def load_model(self, model_path):
        self.model = load_model(model_path)
        base = os.path.basename(model_path)
        sfile = base.replace('model_', 'scalers_').replace('.h5', '.joblib')
        spath = os.path.join(os.path.dirname(model_path), sfile)
        if os.path.exists(spath):
            self.sensor_scalers = joblib.load(spath)
        else:
            print(f"경고: 스케일러 파일이 없습니다: {spath}")
        return self.model
