import numpy as np
import os
from datetime import datetime
import joblib

import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dense,
    LayerNormalization,
    Add,
    MultiHeadAttention,
    Lambda,
    Dropout,
    Concatenate,
    GlobalAveragePooling1D,
    Embedding,
)

from sklearn.preprocessing import StandardScaler, MinMaxScaler
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
        self.feature_map = {
            "acc_axes": [0, 1, 2],
            "gyro_axes": [3, 4, 5],
            "acc_norm": 6,
            "gyro_norm": 7,
        }
        self.use_concat = True

    def build_model(self):
        inputs = Input(shape=(self.window_size, self.num_features))

        x = LSTM(64, return_sequences=True)(inputs)
        x = LSTM(32, return_sequences=False)(x)

        outputs = Dense(2)(x)

        self.model = tf.keras.Model(inputs, outputs)

        self.model.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss="mse",
            metrics=["mae", Huber(delta=1.0)],
        )
        return self.model

    # def build_model(self):
    #     inputs = Input(shape=(self.window_size, self.num_features))  # (B,T,F)

    #     # 기본 LSTM 스택 (기존과 동일)
    #     x = LSTM(64, return_sequences=True)(inputs)
    #     x = LSTM(32, return_sequences=True)(x)

    #     attn = MultiHeadAttention(num_heads=2, key_dim=16)(x, x)
    #     h = Concatenate(axis=-1)([x, attn])
    #     h = GlobalAveragePooling1D()(h)
    #     outputs = Dense(2)(h)

    #     self.model = tf.keras.Model(inputs, outputs)
    #     self.model.compile(
    #         optimizer=Adam(learning_rate=1e-3), loss="mse", metrics=["mae", Huber()]
    #     )
    #     return self.model
    
    # def build_model(
    #     self,
    #     d_model: int = 128,         # 임베딩 차원
    #     num_heads: int = 4,         # 멀티헤드 수
    #     ff_dim: int = 256,          # FFN 내부 차원
    #     num_layers: int = 3,        # 인코더 블록 수
    #     dropout: float = 0.1
    # ):
    #     """
    #     기본 트랜스포머 인코더로 시계열을 처리하고, GlobalAvgPool 후 2차원 회귀 출력.
    #     입력: (batch, T=window_size, F=num_features)
    #     출력: (batch, 2)  -> [speed, d_heading] (MinMax 스케일 공간)
    #     """
    #     assert self.num_features is not None, "num_features가 None입니다. train_model에서 입력으로 동기화됩니다."

    #     inputs = Input(shape=(self.window_size, self.num_features))   # (B, T, F)

    #     # 1) 선형 프로젝션(피처 -> d_model)
    #     x = Dense(d_model)(inputs)   # (B, T, d_model)

    #     # 2) 학습 가능한 위치 임베딩 추가
    #     pos_idx = tf.range(self.window_size)                           # (T,)
    #     pos_emb = Embedding(input_dim=self.window_size, output_dim=d_model, name="pos_embedding")(pos_idx)  # (T, d_model)
    #     x = x + tf.expand_dims(pos_emb, axis=0)                        # broadcast to (B, T, d_model)
    #     x = Dropout(dropout)(x)

    #     # 3) 트랜스포머 인코더 블록들
    #     key_dim = max(1, d_model // num_heads)
    #     for _ in range(num_layers):
    #         # (a) Self-Attention + Residual + LN
    #         attn_out = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout)(x, x)
    #         x = LayerNormalization(epsilon=1e-6)(x + attn_out)
    #         # (b) FFN + Residual + LN
    #         f = Dense(ff_dim, activation="gelu")(x)
    #         f = Dropout(dropout)(f)
    #         f = Dense(d_model)(f)
    #         x = LayerNormalization(epsilon=1e-6)(x + f)

    #     # 4) 시퀀스 풀링 후 회귀 헤드
    #     x_last = Lambda(lambda t: t[:, -1, :], name="last_timestep")(x)  # (B, d_model)
    #     outputs = Dense(2)(x_last)                  # [speed, d_heading]

    #     self.model = Model(inputs, outputs)
    #     # self.model.compile(optimizer=Adam(5e-4), loss=Huber(delta=1.3), metrics=["mae"])
    #     self.model.compile(optimizer=Adam(5e-4), loss='mae', metrics=["mse", Huber(delta=1.3)])
    #     return self.model
    
    def scale_sensor_data(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        - acc 3축, gyro 3축을 각각 '그룹'으로 묶어 하나의 μ,σ(스칼라)로 표준화
        - acc_norm, gyro_norm도 기존 값 그대로 개별 스케일링
        """
        if X.ndim != 3:
            raise ValueError("X must be 3D: (N, T, F)")
        N, T, F = X.shape
        eps = 1e-8

        X2d = X.reshape(-1, F).astype(np.float32)
        out = X2d.copy()

        if fit:
            self.sensor_scalers = {}

        acc_axes = self.feature_map.get("acc_axes", [])
        gyro_axes = self.feature_map.get("gyro_axes", [])
        acc_norm_i = self.feature_map.get("acc_norm", None)
        gyro_norm_i = self.feature_map.get("gyro_norm", None)

        # --- (A) 그룹 표준화 (acc, gyro 3축 각각 하나의 μ, σ) ---
        if acc_axes:
            if fit:
                block = X2d[:, acc_axes]
                mu = float(block.mean())
                sigma = float(block.std()) + eps
                self.sensor_scalers["acc_group"] = {
                    "mu": mu,
                    "sigma": sigma,
                    "idxs": acc_axes,
                }
            p = self.sensor_scalers["acc_group"]
            out[:, acc_axes] = (out[:, acc_axes] - p["mu"]) / p["sigma"]

        if gyro_axes:
            if fit:
                block = X2d[:, gyro_axes]
                mu = float(block.mean())
                sigma = float(block.std()) + eps
                self.sensor_scalers["gyro_group"] = {
                    "mu": mu,
                    "sigma": sigma,
                    "idxs": gyro_axes,
                }
            p = self.sensor_scalers["gyro_group"]
            out[:, gyro_axes] = (out[:, gyro_axes] - p["mu"]) / p["sigma"]

        # --- (B) Norm 채널도 개별 스케일링 ---
        if acc_norm_i is not None:
            if fit:
                mu = float(X2d[:, acc_norm_i].mean())
                sigma = float(X2d[:, acc_norm_i].std()) + eps
                self.sensor_scalers["acc_norm"] = {
                    "mu": mu,
                    "sigma": sigma,
                    "idx": acc_norm_i,
                }
            p = self.sensor_scalers["acc_norm"]
            out[:, acc_norm_i] = (out[:, acc_norm_i] - p["mu"]) / p["sigma"]

        if gyro_norm_i is not None:
            if fit:
                mu = float(X2d[:, gyro_norm_i].mean())
                sigma = float(X2d[:, gyro_norm_i].std()) + eps
                self.sensor_scalers["gyro_norm"] = {
                    "mu": mu,
                    "sigma": sigma,
                    "idx": gyro_norm_i,
                }
            p = self.sensor_scalers["gyro_norm"]
            out[:, gyro_norm_i] = (out[:, gyro_norm_i] - p["mu"]) / p["sigma"]

        return out.reshape(N, T, F).astype(np.float32)

    def train_model(self, X_train, Y_train, X_val, Y_val):
        X_tr, X_te, Y_tr, Y_te = X_train, X_val, Y_train, Y_val

        self.num_features = X_tr.shape[2]

        X_tr_s = self.scale_sensor_data(X_tr, fit=True)
        X_te_s = self.scale_sensor_data(X_te, fit=False)

        # self.y_speed_scaler = StandardScaler()
        # self.y_hc_scaler = StandardScaler()
        self.y_speed_scaler = MinMaxScaler()
        self.y_hc_scaler    = MinMaxScaler(feature_range=(-1, 1))

        y1 = self.y_speed_scaler.fit_transform(Y_tr[:, :1])
        y2 = self.y_hc_scaler.fit_transform(Y_tr[:, 1:2])
        Y_tr_s = np.hstack([y1, y2]).astype(np.float32)

        y1_te = self.y_speed_scaler.transform(Y_te[:, :1])
        y2_te = self.y_hc_scaler.transform(Y_te[:, 1:2])
        Y_te_s = np.hstack([y1_te, y2_te]).astype(np.float32)

        self.build_model()

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=10,
                min_delta=5e-4,
                restore_best_weights=True,
            ),
            # ReduceLROnPlateau(
            #     monitor="val_loss",
            #     factor=0.5,
            #     patience=3,
            #     min_lr=1e-6,
            #     cooldown=1,
            #     verbose=1,
            # ),
            ModelCheckpoint(
                "best_model.h5", monitor="val_loss", save_best_only=True, verbose=0
            ),
        ]

        history = self.model.fit(
            X_tr_s,
            Y_tr_s,
            validation_data=(X_te_s, Y_te_s),
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=callbacks,
            verbose=1,
        )
        return history

    def save_model(self, model_dir="saved_models"):
        os.makedirs(model_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"ws{self.window_size}_{ts}"
        mpath = os.path.join(model_dir, f"{base}.h5")
        spath = os.path.join(model_dir, f"{base}.joblib")

        self.model.save(mpath)
        joblib.dump(
            {
                "sensor": self.sensor_scalers,
                "y_speed": self.y_speed_scaler,
                "y_hc": self.y_hc_scaler,
            },
            spath,
        )
        return mpath

    def load_model(self, model_path):
        self.model = load_model(model_path, compile=False)
        spath = model_path.replace(".h5", ".joblib")
        if os.path.exists(spath):
            data = joblib.load(spath)
            self.sensor_scalers = data["sensor"]
            self.y_speed_scaler = data["y_speed"]
            self.y_hc_scaler = data["y_hc"]
        # 필요한 경우 여기서 다시 compile (옵션)
        self.model.compile(
            optimizer=Adam(1e-3, clipnorm=1.0), loss=Huber(delta=1.0), metrics=["mae"]
        )
        return self.model

    def plot_training_history(self, history):
        plt.figure(figsize=(8, 5))
        epochs = range(1, len(history.history["loss"]) + 1)
        plt.plot(epochs, history.history["loss"], label="Train Loss")
        plt.plot(epochs, history.history["val_loss"], label="Val Loss")
        plt.xlabel("Epochs")
        plt.ylabel("MSE Loss")
        plt.title("Training & Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.show()


