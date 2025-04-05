import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from datetime import datetime
import pandas as pd

class ModelTrainer:
    """
    LSTM 모델의 구축, 학습 및 저장을 담당하는 클래스.
    
    주요 기능:
      - 모델 구조 구성
      - 데이터 스케일링 (센서별로)
      - 학습/테스트 데이터 분할 후 모델 학습
      - 학습 결과 시각화 및 모델 저장
    """
    def __init__(self, window_size, num_features, epochs=100, batch_size=128):
        self.window_size = window_size
        self.num_features = num_features
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        # 각 센서 그룹별 MinMaxScaler (범위: -1 ~ 1)
        self.scaler_acc = MinMaxScaler(feature_range=(-1, 1))
        self.scaler_gyro = MinMaxScaler(feature_range=(-1, 1))
        self.scaler_ori = MinMaxScaler(feature_range=(-1, 1))

    def build_model(self):
        """
        LSTM 기반 모델을 구성합니다.
        """
        self.model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(self.window_size, self.num_features)),
            LSTM(64, return_sequences=True),
            LSTM(32, return_sequences=False),
            Dense(2)  # 출력: [속도, 헤딩 변화량]
        ])
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return self.model

    def analyze_scaling(self, X, X_scaled):
        """
        스케일링 전/후의 데이터를 분석하고 시각화합니다.
        
        Args:
            X: 원본 데이터
            X_scaled: 스케일링된 데이터
        """
        # 각 센서별 축 이름
        axes = ['x', 'y', 'z']
        sensors = {
            'Accelerometer': (0, 3),
            'Gyroscope': (3, 6),
            'Orientation': (6, 9)
        }
        
        # 통계 정보 출력
        print("\n=== 스케일링 전/후 통계 정보 ===")
        for sensor_name, (start, end) in sensors.items():
            print(f"\n{sensor_name}:")
            for i, axis in enumerate(axes):
                original = X[:, :, start+i].flatten()
                scaled = X_scaled[:, :, start+i].flatten()
                
                stats = pd.DataFrame({
                    'Original': {
                        'Min': np.min(original),
                        'Max': np.max(original),
                        'Mean': np.mean(original),
                        'Std': np.std(original)
                    },
                    'Scaled': {
                        'Min': np.min(scaled),
                        'Max': np.max(scaled),
                        'Mean': np.mean(scaled),
                        'Std': np.std(scaled)
                    }
                })
                print(f"\n{axis}-axis:")
                print(stats)
        
        # 시각화
        plt.figure(figsize=(15, 10))
        
        for i, (sensor_name, (start, end)) in enumerate(sensors.items()):
            for j, axis in enumerate(axes):
                # 히스토그램
                plt.subplot(3, 3, i*3 + j + 1)
                plt.hist(X[:, :, start+j].flatten(), bins=50, alpha=0.5, label='Original')
                plt.hist(X_scaled[:, :, start+j].flatten(), bins=50, alpha=0.5, label='Scaled')
                plt.title(f'{sensor_name} - {axis}-axis')
                plt.legend()
        
        plt.tight_layout()
        plt.show()

    def scale_sensor_data(self, X):
        """
        각 센서별로 개별적인 스케일러를 사용
        각 센서에서 각 축별로 스케일링
        """
        total_samples, win_size, _ = X.shape
        X_original = X.copy()  # 원본 데이터 보존

        # Accelerometer 스케일링
        X_acc = X[:, :, 0:3].reshape(-1, 3)
        X_acc_scaled = self.scaler_acc.fit_transform(X_acc)
        X[:, :, 0:3] = X_acc_scaled.reshape(total_samples, win_size, 3)

        # Gyroscope 스케일링
        X_gyro = X[:, :, 3:6].reshape(-1, 3)
        X_gyro_scaled = self.scaler_gyro.fit_transform(X_gyro)
        X[:, :, 3:6] = X_gyro_scaled.reshape(total_samples, win_size, 3)

        # Orientation 스케일링
        X_ori = X[:, :, 6:9].reshape(-1, 3)
        X_ori_scaled = self.scaler_ori.fit_transform(X_ori)
        X[:, :, 6:9] = X_ori_scaled.reshape(total_samples, win_size, 3)

        # 스케일링 분석
        self.analyze_scaling(X_original, X)

        return X

    def train_model(self, X, Y):
        """
        데이터를 학습/테스트 세트로 분리하고, 모델을 학습시킵니다.
        """
        X = self.scale_sensor_data(X)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=41)

        # TensorFlow 최적화를 위해 float32 타입으로 변환
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
        Y_train = Y_train.astype(np.float32)
        Y_test = Y_test.astype(np.float32)

        self.build_model()
        history = self.model.fit(
            X_train, Y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(X_test, Y_test),
            verbose=1
        )
        return history, (X_train, Y_train, X_test, Y_test)

    def save_model(self, save_dir='saved_models'):
        """
        학습된 모델을 지정 폴더에 저장합니다.
        """
        os.makedirs(save_dir, exist_ok=True)
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        model_filename = f"model_{timestamp}.h5"
        model_path = os.path.join(save_dir, model_filename)
        self.model.save(model_path)
        return model_path

    def plot_training_history(self, history):
        """
        학습 및 검증 손실을 그래프로 시각화합니다.
        """
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
