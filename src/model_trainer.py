import numpy as np
import os
from datetime import datetime
import joblib

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, MultiHeadAttention, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization, Dropout, Add, Input, GlobalAveragePooling1D, SpatialDropout1D

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class ModelTrainer:
    """
    - scale_sensor_data: 윈도우 전체 채널마다 MinMaxScaler 적용
    - build_model: self.num_features 입력 → LSTM 모델
    - train_model: train/test split → scaling → Y 스케일링 → 학습
    """

    def __init__(self, window_size=50, num_features=None, epochs=100, batch_size=128):
        self.window_size  = window_size
        self.num_features = num_features
        self.epochs       = epochs
        self.batch_size   = batch_size

        self.sensor_scalers = {}  # idx→MinMaxScaler
        self.y_speed_scaler = None
        self.y_hc_scaler    = None
        self.model          = None

    # 그나마 좀 되는 코드 
    def build_model(self): 
        if self.num_features is None: 
            raise ValueError("num_features를 지정하거나, train_model()에서 자동 설정하세요.") 
        
        inputs = tf.keras.Input(shape=(self.window_size, self.num_features)) # 1) 첫 LSTM: 시퀀스 출력 
        x = LSTM(128, return_sequences=True)(inputs) 
        x = Dropout(0.2)(x) # 2) (중간) Self-Attention 블록 + Residual 연결 

        x = LayerNormalization()(x)
        attn = MultiHeadAttention(num_heads=2, key_dim=32)(x, x) 
        x = Add()([x, attn]) 
        x = Dropout(0.2)(x)

        x = LSTM(128, return_sequences=True)(x) 
        x = Dropout(0.2)(x) 
        
        x = LSTM(64, return_sequences=False)(x) 
        x = Dropout(0.2)(x) # 4) 출력 
        
        outputs = Dense(2, name='predictions')(x) 
        
        self.model = tf.keras.Model(
            inputs=inputs, 
            outputs=outputs, 
            name='lstm_attn_lstm') 
        
        self.model.compile( 
            optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4), 
            loss=tf.keras.losses.Huber(delta=1.0),
            metrics=['mse'] ) 
        
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
                #scaler = RobustScaler()
                flat_s = scaler.fit_transform(col)
                self.sensor_scalers[idx] = scaler
            else:
                flat_s = self.sensor_scalers[idx].transform(col)
            scaled[:, idx] = flat_s.ravel()

        return scaled.reshape(n_s, w, f)

    def train_model_grouped(self, X: np.ndarray, Y: np.ndarray, groups: np.ndarray, val_ratio: float = 0.2):
        """
        groups: 각 샘플이 어느 '파일'에서 왔는지 나타내는 동일 길이의 벡터 (예: 파일 인덱스)
                같은 그룹의 샘플은 학습/검증에 동시에 등장하지 않도록 분리.
        """
        # 1) 그룹 단위 분할
        uniq = np.unique(groups)
        n_val = max(1, int(len(uniq) * val_ratio))
        rng = np.random.RandomState(1217)
        rng.shuffle(uniq)

        val_groups = set(uniq[:n_val])
        train_mask = ~np.isin(groups, list(val_groups))
        val_mask   =  np.isin(groups, list(val_groups))

        X_tr, X_te = X[train_mask], X[val_mask]
        Y_tr, Y_te = Y[train_mask], Y[val_mask]

        # 자동 num_features 설정
        self.num_features = X_tr.shape[2]

        # 2) 입력 스케일링
        X_tr_s = self.scale_sensor_data(X_tr, fit=True)
        X_te_s = self.scale_sensor_data(X_te, fit=False)

        # 3) Y 스케일링 (속도, heading-change 따로)
        self.y_speed_scaler = StandardScaler()
        self.y_hc_scaler    = StandardScaler()

        y1 = self.y_speed_scaler.fit_transform(Y_tr[:, :1])
        y2 = self.y_hc_scaler.fit_transform(Y_tr[:, 1:2])
        Y_tr_s = np.hstack([y1, y2]).astype(np.float32)

        y1_te = self.y_speed_scaler.transform(Y_te[:, :1])
        y2_te = self.y_hc_scaler.transform(Y_te[:, 1:2])
        Y_te_s = np.hstack([y1_te, y2_te]).astype(np.float32)

        # 4) 모델 생성
        self.build_model()

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, min_delta=1e-4),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-5),
            ModelCheckpoint("best_grouped_model.h5", monitor="val_loss", save_best_only=True, verbose=0),
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




    # def train_model(self, X: np.ndarray, Y: np.ndarray):
    #     # 1) split
    #     X_tr, X_te, Y_tr, Y_te = train_test_split(X, Y, test_size=0.2, random_state=1217)

    #     # 자동 num_features 설정
    #     self.num_features = X_tr.shape[2]

    #     # 2) X 스케일
    #     X_tr_s = self.scale_sensor_data(X_tr, fit=True)
    #     X_te_s = self.scale_sensor_data(X_te, fit=False)

    #     # 3) Y 스케일
    #     self.y_speed_scaler = MinMaxScaler(feature_range=(-1, 1))
    #     self.y_hc_scaler    = MinMaxScaler(feature_range=(-1, 1))
        
    #     # self.y_speed_scaler = RobustScaler()
    #     # self.y_hc_scaler    = RobustScaler()
        
        

    #     y1 = self.y_speed_scaler.fit_transform(Y_tr[:, :1])
    #     y2 = self.y_hc_scaler.fit_transform(Y_tr[:, 1:2])
    #     Y_tr_s = np.hstack([y1, y2]).astype(np.float32)

    #     y1_te = self.y_speed_scaler.transform(Y_te[:, :1])
    #     y2_te = self.y_hc_scaler.transform(Y_te[:, 1:2])
    #     Y_te_s = np.hstack([y1_te, y2_te]).astype(np.float32)

    #     # 4) 모델 구성 & 학습
    #     self.build_model()
    #     callbacks = [
    #         EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
    #         ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10)
    #     ]
    #     history = self.model.fit(
    #         X_tr_s, Y_tr_s,
    #         validation_data=(X_te_s, Y_te_s),
    #         batch_size=self.batch_size,
    #         epochs=self.epochs,
    #         callbacks=callbacks,
    #         verbose=1
    #     )
    #     return history

    def save_model(self, model_dir='saved_models'):
        os.makedirs(model_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        mpath = os.path.join(model_dir, f'model_{ts}.h5')
        spath = os.path.join(model_dir, f'scalers_{ts}.joblib')

        # 1) 모델 저장
        self.model.save(mpath)
        # 2) 스케일러 저장
        joblib.dump({
            'sensor': self.sensor_scalers,
            'y_speed': self.y_speed_scaler,
            'y_hc':    self.y_hc_scaler
        }, spath)

        return mpath

    def load_model(self, model_path):
        self.model = load_model(model_path, compile=False)
        spath = model_path.replace('model_', 'scalers_').replace('.h5', '.joblib')
        if os.path.exists(spath):
            data = joblib.load(spath)
            self.sensor_scalers = data['sensor']
            self.y_speed_scaler = data['y_speed']
            self.y_hc_scaler    = data['y_hc']
        return self.model
    
    def plot_training_history(self, history):
        """
        Train/Validation Loss 
        """
        plt.figure(figsize=(8, 5))
        epochs = range(1, len(history.history['loss']) + 1)
        plt.plot(epochs, history.history['loss'],     label='Train Loss')
        plt.plot(epochs, history.history['val_loss'], label='Val Loss')
        plt.xlabel('Epochs')
        plt.ylabel('MSE Loss')
        plt.title('Training & Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

