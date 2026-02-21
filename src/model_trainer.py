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
    Input, Conv1D, Add, Activation, Dropout, LSTM, 
    LayerNormalization, GlobalAveragePooling1D, Dense
)
from tensorflow.keras import regularizers
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# def attention_pooling(x):
#     """
#     x: (B, T, D)
#     returns: (B, D)  (time-weighted sum)
#     """
#     # (B, T, 1) : 각 timestep 중요도 점수
#     scores = Dense(1, name="attn_pool_score")(x)

#     # (B, T, 1) : time 축으로 softmax -> 가중치
#     weights = Softmax(axis=1, name="attn_pool_weights")(scores)

#     # (B, T, D) : 가중치 적용
#     weighted = Multiply(name="attn_pool_apply")([x, weights])

#     # (B, D) : time 축으로 가중합
#     pooled = Lambda(lambda t: tf.reduce_sum(t, axis=1), name="attn_pool_sum")(weighted)
#     return pooled

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
        self.x_scaler = None
        self.y_scaler = None
        self.model = None
        
    # def build_model(self):
    #     inputs = Input(shape=(self.window_size, self.num_features))  # (B,T,F)

    #     x = LSTM(128, return_sequences=True)(inputs)
    #     x = LSTM(64, return_sequences=True)(x)

    #     attn = MultiHeadAttention(num_heads=2, key_dim=32)(x, x)
    #     x = Add()([x, attn])
    #     # x = LayerNormalization()(x)
    #     # x = Dropout(0.1)(x)

    #     x = attention_pooling(x)

    #     outputs = Dense(2)(x)

    #     self.model = tf.keras.Model(inputs, outputs)
    #     self.model.compile(
    #         optimizer=Adam(learning_rate=1e-3),
    #         loss="mae"
    #     )
    #     return self.model
    
    
    def build_model(self):
        inputs = Input(shape=(self.window_size, self.num_features))
    
        x = LSTM(256, return_sequences=True)(inputs)
        x = LSTM(128, return_sequences=False)(x)
        
        outputs = Dense(2)(x)

        self.model = tf.keras.Model(inputs, outputs)

        self.model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss="mse",
            #metrics=["mse", Huber(delta=1.0)],
        )
        return self.model
    

    
    def train_model(self, X, Y, test_size=0.2, random_state=42):

        # -----------------------------
        # 1. Train / Validation split
        # -----------------------------
        X_tr, X_te, Y_tr, Y_te = train_test_split(
            X, Y,
            test_size=test_size,
            random_state=random_state,
            shuffle=True
        )
        
        def plot_y_distribution(Y, title="Y Distribution"):
            speed = Y[:, 0]
            dh = Y[:, 1]

            plt.figure(figsize=(12,5))

            plt.subplot(1,2,1)
            plt.hist(speed, bins=50)
            plt.title(f"{title} - Speed/Disp")
            plt.xlabel("Value")
            plt.ylabel("Count")

            plt.subplot(1,2,2)
            plt.hist(np.degrees(dh), bins=50)
            plt.title(f"{title} - Heading Change")
            plt.xlabel("Value")
            plt.ylabel("Count")

            plt.tight_layout()
            plt.show()


        plot_y_distribution(Y_tr, "Train")
        plot_y_distribution(Y_te, "Test")
        
        self.num_features = X_tr.shape[2]
        
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        
        X_tr_2d = X_tr.reshape(-1, self.num_features)
        x_te_2d = X_te.reshape(-1, self.num_features)
        
        X_tr_2d_s = self.x_scaler.fit_transform(X_tr_2d)
        x_te_2d_s = self.x_scaler.transform(x_te_2d)
        
        X_tr_s = X_tr_2d_s.reshape(X_tr.shape)
        X_te_s = x_te_2d_s.reshape(X_te.shape)
        
        Y_tr_s = self.y_scaler.fit_transform(Y_tr)
        Y_te_s = self.y_scaler.transform(Y_te)

        self.build_model()

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=5,
                min_delta=5e-4,
                restore_best_weights=False,
            ),
        ]

        history = self.model.fit(
            X_tr_s,
            Y_tr_s,
            validation_data=(X_te_s, Y_te_s),
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=callbacks,
            shuffle=True,
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
                "x_scaler": self.x_scaler,
                "y_scaler": self.y_scaler,
            },
            spath,
        )
        return mpath

    def load_model(self, model_path):
        self.model = load_model(model_path, compile=False)

        spath = model_path.replace(".h5", ".joblib")

        if os.path.exists(spath):
            data = joblib.load(spath)
            self.x_scaler = data["x_scaler"]
            self.y_scaler = data["y_scaler"]

        self.model.compile(
            optimizer=Adam(1e-3),
            loss="mae",
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

