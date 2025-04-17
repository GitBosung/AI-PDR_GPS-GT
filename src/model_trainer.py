import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization, Dropout
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
        self.model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(self.window_size, self.num_features)),
            
            LSTM(64, return_sequences=True),
            
            LSTM(32, return_sequences=False),
            
            Dense(2)  # [속도, 헤딩 변화량]
        ])
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
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
                    'Original': {
                        'Min': np.min(original),
                        'Max': np.max(original),
                        'Mean': np.mean(original),
                        'Std': np.std(original),
                    },
                    'Scaled': {
                        'Min': np.min(scaled),
                        'Max': np.max(scaled),
                        'Mean': np.mean(scaled),
                        'Std': np.std(scaled),
                    }
                }

            stats = pd.DataFrame.from_dict(stats_dict, orient='index')
            print(stats)

        total_plots = sum(end - start for sensor_name, (start, end) in sensors.items())
        n_cols = 4
        n_rows = math.ceil(total_plots / n_cols)

        plt.figure(figsize=(15, 15))
        subplot_index = 1
        for sensor_name, (start, end) in sensors.items():
            for i in range(start, end):
                plt.subplot(n_rows, n_cols, subplot_index)
                subplot_index += 1
                plt.hist(X[:, :, i].flatten(), bins=50, alpha=0.5, label='Original')
                plt.hist(X_scaled[:, :, i].flatten(), bins=50, alpha=0.5, label='Scaled')
                plt.title(f'{sensor_name} index {i}')
                plt.legend()
        plt.tight_layout()
        plt.show()

    def scale_sensor_data(self, X):
        total_samples, win_size, num_features = X.shape
        X_original = X.copy()
        X_scaled = np.zeros_like(X)

        # 기존에 스케일링 할 피처: Accelerometer, Gyroscope, Acc_Norm (총 7 피처)
        axis_indices = {
            'Accelerometer x': 0,
            'Accelerometer y': 1,
            'Accelerometer z': 2,
            'Gyroscope x': 3,
            'Gyroscope y': 4,
            'Gyroscope z': 5,
            'Acc_Norm': 6
        }

        for axis_name, idx in axis_indices.items():
            axis_data = X[:, :, idx].reshape(-1, 1)
            scaler = MinMaxScaler(feature_range=(-1, 1))  # MinMaxScaler 적용
            axis_scaled = scaler.fit_transform(axis_data)
            X_scaled[:, :, idx] = axis_scaled.reshape(total_samples, win_size)
            self.sensor_scalers[axis_name] = scaler

        # Orientation 관련 피처 (cos_roll, sin_roll, ...)는 스케일링 없이 그대로 복사 (인덱스 7부터 시작)
        if num_features > 7:
            X_scaled[:, :, 7:] = X_original[:, :, 7:]
        
        self.analyze_scaling(X_original, X_scaled)
        return X_scaled

    def train_model(self, X, Y):
        X = self.scale_sensor_data(X)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1217)
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
        Y_train = Y_train.astype(np.float32)
        Y_test = Y_test.astype(np.float32)
        self.build_model()
        history = self.model.fit(X_train, Y_train, batch_size=self.batch_size, epochs=self.epochs,
                                 validation_data=(X_test, Y_test), verbose=1)
        return history, (X_train, Y_train, X_test, Y_test)

    def plot_training_history(self, history):
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs_range = range(1, len(train_loss) + 1)
        plt.figure(figsize=(8, 5))
        plt.plot(epochs_range, train_loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training & Validation Loss')
        plt.legend()
        plt.grid()
        plt.show()

    def save_model(self, model_dir='saved_models'):
        os.makedirs(model_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = os.path.join(model_dir, f'model_{timestamp}.h5')
        scaler_path = os.path.join(model_dir, f'scalers_{timestamp}.joblib')

        self.model.save(model_path)
        joblib.dump(self.sensor_scalers, scaler_path)
        return model_path

    def load_model(self, model_path):
        self.model = load_model(model_path)
        model_filename = os.path.basename(model_path)
        scaler_filename = model_filename.replace('model_', 'scalers_').replace('.h5', '.joblib')
        scaler_path = os.path.join(os.path.dirname(model_path), scaler_filename)

        if os.path.exists(scaler_path):
            self.sensor_scalers = joblib.load(scaler_path)
        else:
            print(f"경고: 스케일러 파일을 찾을 수 없습니다: {scaler_path}")
        return self.model
