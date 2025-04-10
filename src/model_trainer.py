import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization, Dropout, Layer
from tensorflow.keras import backend as K
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from datetime import datetime
import pandas as pd
import joblib

class SelfAttention(Layer):
    def __init__(self, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight',
                                shape=(input_shape[-1], input_shape[-1]),
                                initializer='glorot_uniform',
                                trainable=True)
        super(SelfAttention, self).build(input_shape)

    def call(self, x):
        # attention score 계산
        attention = K.dot(x, self.W)
        attention = K.softmax(attention, axis=1)
        
        # attention 가중치 적용
        output = K.sum(attention * x, axis=1)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

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
        self.scaler_acc_norm = MinMaxScaler(feature_range=(-1, 1))

    def build_model(self):
        """
        LSTM 기반 모델을 구성합니다.
        """
        self.model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(self.window_size, self.num_features)),
            LSTM(64, return_sequences=True),
            LSTM(32, return_sequences=True),
            SelfAttention(),
            Dense(2)  # 출력: [속도, Heading Change]
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

        # Acc_Norm 스케일링
        X_acc_norm = X[:, :, 3:4].reshape(-1, 1)
        X_acc_norm_scaled = self.scaler_acc_norm.fit_transform(X_acc_norm)
        X[:, :, 3:4] = X_acc_norm_scaled.reshape(total_samples, win_size, 1)

        # Gyroscope 스케일링
        X_gyro = X[:, :, 4:7].reshape(-1, 3)
        X_gyro_scaled = self.scaler_gyro.fit_transform(X_gyro)
        X[:, :, 4:7] = X_gyro_scaled.reshape(total_samples, win_size, 3)
        
        # Orientation 스케일링
        X_ori = X[:, :, 7:10].reshape(-1, 3)
        X_ori_scaled = self.scaler_ori.fit_transform(X_ori)
        X[:, :, 7:10] = X_ori_scaled.reshape(total_samples, win_size, 3)

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

    def save_model(self, model_dir='saved_models'):
        """
        모델과 스케일러를 저장합니다.
        
        Args:
            model_dir: 모델 저장 디렉토리
            
        Returns:
            저장된 모델 파일 경로
        """
        # 저장 디렉토리 생성
        os.makedirs(model_dir, exist_ok=True)
        
        # 현재 시간을 포함한 파일명 생성
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = os.path.join(model_dir, f'model_{timestamp}.h5')
        scaler_path = os.path.join(model_dir, f'scalers_{timestamp}.joblib')
        
        # 모델 저장
        self.model.save(model_path)
        
        # 스케일러 저장
        scalers = {
            'scaler_acc': self.scaler_acc,
            'scaler_gyro': self.scaler_gyro,
            'scaler_ori': self.scaler_ori,
            'scaler_acc_norm': self.scaler_acc_norm
        }
        joblib.dump(scalers, scaler_path)
        
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

    def load_model(self, model_path):
        """
        모델과 스케일러를 로드합니다.
        
        Args:
            model_path: 모델 파일 경로
            
        Returns:
            로드된 모델
        """
        # 모델 로드
        model = load_model(model_path, custom_objects={'SelfAttention': SelfAttention})
        
        # 스케일러 로드
        # 모델 파일명에서 타임스탬프를 추출하여 스케일러 파일 경로 생성
        model_filename = os.path.basename(model_path)
        scaler_filename = model_filename.replace('model_', 'scalers_').replace('.h5', '.joblib')
        scaler_path = os.path.join(os.path.dirname(model_path), scaler_filename)
        
        if os.path.exists(scaler_path):
            scalers = joblib.load(scaler_path)
            self.scaler_acc = scalers['scaler_acc']
            self.scaler_gyro = scalers['scaler_gyro']
            self.scaler_ori = scalers['scaler_ori']
            self.scaler_acc_norm = scalers['scaler_acc_norm']
        else:
            print(f"경고: 스케일러 파일을 찾을 수 없습니다: {scaler_path}")
        
        return model
