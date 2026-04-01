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
    LayerNormalization, GlobalAveragePooling1D, Dense, MultiHeadAttention, Multiply, Lambda, Softmax, BatchNormalization
)
from tensorflow.keras import regularizers
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def transformer_encoder(x, head_size, num_heads, ff_dim, dropout=0.1):
    # ------------------------
    # 1. Multi-Head Attention
    # ------------------------
    attn = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=head_size
    )(x, x)

    attn = Dropout(dropout)(attn)
    x = Add()([x, attn])
    x = LayerNormalization()(x)

    # ------------------------
    # 2. Feed Forward Network (핵심)
    # ------------------------
    ff = Dense(ff_dim, activation="relu")(x)
    ff = Dense(x.shape[-1])(ff)

    ff = Dropout(dropout)(ff)
    x = Add()([x, ff])
    x = LayerNormalization()(x)

    return x

def weighted_mae_dh(y_true, y_pred):
    error = tf.abs(y_true - y_pred)

    # 🔥 핵심: 회전 클수록 weight 증가
    weight = 1.0 + 5.0 * tf.abs(y_true)   # scale 조절 가능

    return tf.reduce_mean(error * weight)

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
    #     inputs = Input(shape=(self.window_size, self.num_features))

    #     # ------------------------
    #     # 1. LSTM (low-level temporal)
    #     # ------------------------
    #     x = LSTM(128, return_sequences=True)(inputs)

    #     # ------------------------
    #     # 2. Transformer Encoder (핵심)
    #     # ------------------------
    #     x = transformer_encoder(x, head_size=32, num_heads=4, ff_dim=128)
    #     x = transformer_encoder(x, head_size=32, num_heads=4, ff_dim=128)

    #     # ------------------------
    #     # 3. Global pooling
    #     # ------------------------
    #     x = GlobalAveragePooling1D()(x)

    #     # ------------------------
    #     # 4. Output branches
    #     # ------------------------
    #     speed_out = Dense(1, name="speed")(x)
    #     dh_out = Dense(1, name="dh")(x)

    #     self.model = Model(inputs, [speed_out, dh_out])

    #     self.model.compile(
    #         optimizer=Adam(learning_rate=3e-4),  # 약간 낮추는게 좋음
    #         loss={"speed": "mae", "dh": "mae"},
    #         loss_weights={"speed": 1.0, "dh": 10.0},
    #     )

    #     return self.model
    
    def build_model(self):
        inputs = Input(shape=(self.window_size, self.num_features))
    
        x = LSTM(64, return_sequences=True)(inputs)   
        x = LSTM(32, return_sequences=False)(x)
        
        speed_out = Dense(1, name="speed")(x)
        dh_out = Dense(1, name="dh")(x)

        self.model = tf.keras.Model(inputs, [speed_out, dh_out])

        self.model.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss={"speed": "mae", "dh": "mae"},
            loss_weights={"speed": 1.0, "dh": 10.0},  
            #metrics={"speed": ["mae"], "dh": ["mae"]},
        )
        return self.model
    

    # def build_model(self):
    #     inputs = Input(shape=(self.window_size, self.num_features))


    #     x = LSTM(128, return_sequences=True)(inputs)
    #     x = LayerNormalization()(x)
        
    #     x = LSTM(128, return_sequences=True)(x)   
    #     x = LayerNormalization()(x)


    #     attn_out = MultiHeadAttention(
    #         num_heads=4,
    #         key_dim=32
    #     )(x, x)


    #     x = Add()([x, attn_out])
    #     x = LayerNormalization()(x)

    #     x = GlobalAveragePooling1D()(x)

    
    #     speed_branch = Dense(64, activation="relu")(x)
    #     speed_out = Dense(1, name="speed")(speed_branch)
        
    #     dh_branch = Dense(64, activation="relu")(x)
    #     dh_out = Dense(1, name="dh")(dh_branch)

    #     self.model = Model(inputs, [speed_out, dh_out])

    #     self.model.compile(
    #         optimizer=Adam(learning_rate=5e-4),
    #         loss={"speed": "mae", "dh": "mae"},
    #         loss_weights={"speed": 1.0, "dh": 5.0},
    #     )

    #     return self.model
        
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
        
        # ✅ 2-head용 dict 타깃으로 분리 (스케일링 이후 분리)
        Y_tr_dict = {"speed": Y_tr_s[:, 0:1], "dh": Y_tr_s[:, 1:2]}
        Y_te_dict = {"speed": Y_te_s[:, 0:1], "dh": Y_te_s[:, 1:2]}

        self.build_model()

        callbacks = [
            # EarlyStopping(
            #     monitor="val_loss",
            #     patience=5,
            #     min_delta=5e-4,
            #     restore_best_weights=False,
            # ),
        ]

        history = self.model.fit(
            X_tr_s,
            Y_tr_dict,
            validation_data=(X_te_s, Y_te_dict),
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
        h = history.history
        epochs = range(1, len(h["loss"]) + 1)

        # -----------------------------
        # 1) Total loss (optional)
        # -----------------------------
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, h["loss"], label="Train total loss")
        plt.plot(epochs, h["val_loss"], label="Val total loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MAE)")
        plt.title("Total Loss")
        plt.grid(True)
        plt.legend()
        plt.show()

        # -----------------------------
        # 2) Speed head: loss / mae
        # -----------------------------
        plt.figure(figsize=(8, 5))
        if "speed_loss" in h:
            plt.plot(epochs, h["speed_loss"], label="Train speed_loss")
        if "val_speed_loss" in h:
            plt.plot(epochs, h["val_speed_loss"], label="Val speed_loss")
        if "speed_mae" in h:
            plt.plot(epochs, h["speed_mae"], label="Train speed_mae", linestyle="--")
        if "val_speed_mae" in h:
            plt.plot(epochs, h["val_speed_mae"], label="Val speed_mae", linestyle="--")

        plt.xlabel("Epoch")
        plt.ylabel("MAE")
        plt.title("Speed Head (loss / mae)")
        plt.grid(True)
        plt.legend()
        plt.show()

        # -----------------------------
        # 3) DH head: loss / mae
        # -----------------------------
        plt.figure(figsize=(8, 5))
        if "dh_loss" in h:
            plt.plot(epochs, h["dh_loss"], label="Train dh_loss")
        if "val_dh_loss" in h:
            plt.plot(epochs, h["val_dh_loss"], label="Val dh_loss")
        if "dh_mae" in h:
            plt.plot(epochs, h["dh_mae"], label="Train dh_mae", linestyle="--")
        if "val_dh_mae" in h:
            plt.plot(epochs, h["val_dh_mae"], label="Val dh_mae", linestyle="--")

        plt.xlabel("Epoch")
        plt.ylabel("MAE")
        plt.title("Heading Change Head (loss / mae)")
        plt.grid(True)
        plt.legend()
        plt.show()

