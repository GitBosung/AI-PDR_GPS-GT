# src/data_processor.py

import numpy as np
import pandas as pd
import os
import time
import logging
from datetime import datetime
from pyproj import CRS, Transformer
from scipy.interpolate import PchipInterpolator, UnivariateSpline
from scipy.signal import savgol_filter
import math

logger = logging.getLogger(__name__)

class DataProcessor:
    """
    데이터 로딩 및 전처리, 슬라이딩 윈도우 기반 데이터 생성 등 기능.
    """
    def __init__(self):
        pass

    @staticmethod
    def load_and_preprocess_csv(file_path, delimiter=',', header=0, skiprows=300):
        start_time = time.time()
        logger.info(f"CSV 파일 로딩 시작: {file_path}")

        df = pd.read_csv(file_path, delimiter=delimiter, header=header, skiprows=skiprows)
        if len(df) > 100:
            df = df.iloc[:-100].reset_index(drop=True)
        logger.info(f"CSV 파일 로딩 완료: {len(df)} 행")

        # 컬럼 이름 재정의
        df.columns = [
            'Time',
            'Accelerometer x', 'Accelerometer y', 'Accelerometer z',
            'Gyroscope x', 'Gyroscope y', 'Gyroscope z',
            'Magnetometer x', 'Magnetometer y', 'Magnetometer z',
            'Orientation x', 'Orientation y', 'Orientation z',
            'Pressure', 'Latitude', 'Longitude', 'Altitude', 'Speed_GPS'
        ]

        # 시간 전처리
        df['Time'] = pd.to_datetime(df['Time'], format='%Y-%m-%d %H:%M:%S.%f')
        start_time_dt = df['Time'].iloc[0]
        df['Elapsed Time'] = (df['Time'] - start_time_dt).dt.total_seconds()

        # GPS → ENU 변환
        df = DataProcessor.process_gps_data(df)

        # 벡터 노름 계산
        df['Acc_Norm']  = np.linalg.norm(df[['Accelerometer x','Accelerometer y','Accelerometer z']].values, axis=1)
        df['Gyro_Norm'] = np.linalg.norm(df[['Gyroscope x','Gyroscope y','Gyroscope z']].values, axis=1)

        # 순수 속도·헤딩 변화량 계산
        df = DataProcessor.compute_speed_and_heading(df)

        return df

    @staticmethod
    def process_gps_data(df):
        ref_lat = df['Latitude'].iat[0]
        ref_lon = df['Longitude'].iat[0]
        ref_alt = df['Altitude'].iat[0]

        wgs84  = CRS.from_epsg(4326)
        utm52  = CRS.from_epsg(32652)
        transformer = Transformer.from_crs(wgs84, utm52, always_xy=True)

        xs, ys = transformer.transform(df['Longitude'].values, df['Latitude'].values)
        df['E'] = xs - xs[0]
        df['N'] = ys - ys[0]
        df['U'] = df['Altitude'] - ref_alt

        df['Delta X'] = df['E'].diff().fillna(0)
        df['Delta Y'] = df['N'].diff().fillna(0)
        return df

    @staticmethod
    def compute_speed_and_heading(df, time_interval=1.0):
        df['Delta X'] = df['E'].diff().fillna(0)
        df['Delta Y'] = df['N'].diff().fillna(0)
        df['Delta D'] = np.hypot(df['Delta X'], df['Delta Y'])
        df['Speed']   = df['Delta D']

        move_mask = df['Delta D'] > 0
        df_move   = df[move_mask].copy()

        df_move['Heading'] = np.arctan2(df_move['Delta Y'], df_move['Delta X'])
        df_move['Heading'] = df_move['Heading'].fillna(method='ffill').fillna(0)
        hc = df_move['Heading'].diff().fillna(0)
        df_move['Pure Heading Change'] = ((hc + np.pi) % (2 * np.pi)) - np.pi

        df['Pure Speed']          = 0.0
        df['Pure Heading Change'] = 0.0
        df.loc[df_move.index, 'Pure Speed']          = df_move['Speed']
        df.loc[df_move.index, 'Pure Heading Change'] = df_move['Pure Heading Change']
        df['Pure Heading Change (deg)'] = np.degrees(df['Pure Heading Change'])

        return df

    @staticmethod
    def make_XY_using_dataframe(df, window_size=50, stride=1, freq=50):
        from numpy.lib.stride_tricks import sliding_window_view

        logger.info(f"X, Y 생성 시작 (window={window_size}, stride={stride}, freq={freq})")

        t_full = df['Elapsed Time'].values

        # Speed 보간
        t_spd = df.loc[df['Pure Speed'] != 0, 'Elapsed Time'].values
        v_spd = df.loc[df['Pure Speed'] != 0, 'Pure Speed'].values
        spd_pchip = PchipInterpolator(t_spd, v_spd, extrapolate=False)
        v_full    = np.nan_to_num(spd_pchip(t_full), nan=0.0)

        # Heading Change 보간 (앵커 포함)
        t_hc = df.loc[df['Pure Heading Change'] != 0, 'Elapsed Time'].values
        hc   = df.loc[df['Pure Heading Change'] != 0, 'Pure Heading Change'].values

        t_start, t_end = t_full[0], t_full[-1]
        t_hc_aug = np.concatenate(([t_start], t_hc, [t_end]))
        hc_aug   = np.concatenate(([0.0],   hc,   [0.0]))

        hc_pchip  = UnivariateSpline(t_hc_aug, hc_aug, k=2, s=0.01)
        hc_full   = np.nan_to_num(hc_pchip(t_full), nan=0.0)

        eps, delta = 0.1, 0.1
        weight     = 0.5 * (1 + np.tanh((np.abs(hc_full) - eps) / delta))
        hc_gated   = weight * hc_full

        Y_full = np.stack((v_full, hc_gated), axis=1)

        sensor_cols = [
            'Accelerometer x','Accelerometer y','Accelerometer z',
            'Gyroscope x','Gyroscope y','Gyroscope z',
            'Acc_Norm','Gyro_Norm',
        ]
        sensor_data = df[sensor_cols].values

        min_len     = min(len(sensor_data), len(Y_full))
        sensor_data = sensor_data[:min_len]
        Y_full      = Y_full[:min_len]

        windows = sliding_window_view(
            sensor_data,
            window_shape=(window_size, sensor_data.shape[1])
        ).squeeze(1)
        X = windows[::stride]
        Y = Y_full[window_size - 1 :: stride]

        X = X.astype(np.float32)
        Y = Y.astype(np.float32)
        logger.info(f"X, Y 생성 완료: X={X.shape}, Y={Y.shape}")

        return X, Y
