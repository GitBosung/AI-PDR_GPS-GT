# src/model_trainer.py

import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization, Dropout
from tensorflow.keras.losses import Huber  # Huber Loss import 추가
from sklearn.preprocessing import StandardScaler
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
      - 센서 데이터 스케일링 (10개 피처 전체에 적용)
      - 슬라이딩 윈도우 방식 입력 데이터 생성
      - 학습 손실 시각화 및 모델 저장
    """
    def __init__(self, window_size, num_features, epochs=100, batch_size=128):
        self.window_size = window_size
        self.num_features = num_features  # now 10
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.scaler = StandardScaler()

    def build_model(self):
        """
        LSTM 기반 모델 구성:
          - 입력 shape: (window_size, num_features)
          - 출력: [속도, 헤딩 변화량]
        """
        
        self.model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(self.window_size, self.num_features)),
            LSTM(32, return_sequences=True),
            LSTM(16, return_sequences=False),
            Dense(2)  # [속도, 헤딩 변화량]
        ])
        
        # 기존의 mse loss 대신 Huber loss 사용 (delta 값은 하이퍼파라미터로 설정, 여기서는 예시로 1.0 사용)
        self.model.compile(optimizer='adam', loss=Huber(delta=1.0), metrics=['mae'])
        return self.model

    def analyze_scaling(self, X, X_scaled):
        """
        스케일링 전후 통계 정보 출력 및 센서별 히스토그램 시각화
        """
        axes = ['x', 'y', 'z']
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
                stats = pd.DataFrame({
                    'Original': {
                        'Min': np.nanmin(original), 
                        'Max': np.nanmax(original), 
                        'Mean': np.nanmean(original), 
                        'Std': np.nanstd(original)
                    },
                    'Scaled': {
                        'Min': np.nanmin(scaled), 
                        'Max': np.nanmax(scaled), 
                        'Mean': np.nanmean(scaled), 
                        'Std': np.nanstd(scaled)
                    }
                }, index=[0])
                print(f"Feature index {i}:")
                print(stats)
        
        # 전체 플롯 그리드 생성
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
        """
        피처 전체에 대해 스케일링 적용
        """
        total_samples, win_size, num_features = X.shape
        X_original = X.copy()
        X_reshaped = X.reshape(-1, num_features)
        X_reshaped_scaled = self.scaler.fit_transform(X_reshaped)
        X_scaled = X_reshaped_scaled.reshape(total_samples, win_size, num_features)
        self.analyze_scaling(X_original, X_scaled)
        return X_scaled

    def train_model(self, X, Y):
        """
        학습/테스트 세트 구성 및 모델 학습 진행
        """
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
        """
        학습 및 검증 손실을 에포크별 시각화
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

    def save_model(self, model_dir='saved_models'):
        """
        학습된 모델과 스케일러를 파일로 저장
        """
        os.makedirs(model_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = os.path.join(model_dir, f'model_{timestamp}.h5')
        scaler_path = os.path.join(model_dir, f'scalers_{timestamp}.joblib')
        self.model.save(model_path)
        scalers = {'scaler': self.scaler}
        joblib.dump(scalers, scaler_path)
        return model_path

    def load_model(self, model_path):
        """
        저장된 모델 및 스케일러 로드
        """
        self.model = load_model(model_path)
        model_filename = os.path.basename(model_path)
        scaler_filename = model_filename.replace('model_', 'scalers_').replace('.h5', '.joblib')
        scaler_path = os.path.join(os.path.dirname(model_path), scaler_filename)
        if os.path.exists(scaler_path):
            scalers = joblib.load(scaler_path)
            self.scaler = scalers['scaler']
        else:
            print(f"경고: 스케일러 파일을 찾을 수 없습니다: {scaler_path}")
        return self.model
