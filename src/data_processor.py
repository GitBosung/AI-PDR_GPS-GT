import numpy as np
import pandas as pd
import os
import time
import logging
from datetime import datetime
from pyproj import Proj
from scipy.interpolate import PchipInterpolator, UnivariateSpline
from scipy.signal import savgol_filter
import math

logger = logging.getLogger(__name__)

# 캐시된 UTM zone 52N 변환 객체 (EPSG:32652)
_proj_utm52 = Proj("epsg:32652")

class DataProcessor:
    """
    데이터 로딩 및 전처리, 슬라이딩 윈도우 기반 데이터 생성 등 기능.
    """
    def __init__(self):
        pass
    
    @staticmethod
    def load_and_preprocess_csv(file_path, delimiter=',', header=0, skiprows=250):
        """
        1) 원본 센서·GPS 데이터 로드 및 전처리
        2) 50행 단위로 1 Hz 서브샘플링 (df_1hz)
        3) df_1hz를 기준으로 ENU 변환 및 ΔX/Y 계산
        4) df_1hz를 50 Hz로 선형 보간 → df_50hz (ΔX/Y 포함)
        """
        # --- 1) 원본 로드·전처리 ---
        df = pd.read_csv(file_path, delimiter=delimiter, header=header, skiprows=skiprows)
        if len(df) > 100:
            df = df.iloc[:-100].reset_index(drop=True)

        df.columns = [
            'Time',
            'Accelerometer x', 'Accelerometer y', 'Accelerometer z',
            'Gyroscope x', 'Gyroscope y', 'Gyroscope z',
            'Magnetometer x', 'Magnetometer y', 'Magnetometer z',
            'Orientation x', 'Orientation y', 'Orientation z',
            'Pressure', 'Latitude', 'Longitude', 'Altitude', 'Speed_GPS'
        ]
        df['Time'] = pd.to_datetime(df['Time'], format='%Y-%m-%d %H:%M:%S.%f')
        start_dt = df['Time'].iloc[0]
        df['Elapsed Time'] = (df['Time'] - start_dt).dt.total_seconds()

        # 센서 노름 추가
        df['Acc_Norm']  = np.linalg.norm(df[['Accelerometer x','Accelerometer y','Accelerometer z']], axis=1)
        df['Gyro_Norm'] = np.linalg.norm(df[['Gyroscope x','Gyroscope y','Gyroscope z']], axis=1)

        # --- 2) 1 Hz 서브샘플링 (50행 단위) ---
        df['Chunk'] = (np.arange(len(df)) // 50).astype(int)
        df_1hz = (
            df
            .groupby('Chunk', as_index=False)
            .apply(lambda g: g.iloc[len(g)//2])
            .reset_index(drop=True)
        )

        # --- 3) df_1hz → ENU 변환 & Δ 계산 ---
        # 기존 Transformer 로직 대신 캐시된 Proj 객체 사용
        lon = df_1hz['Longitude'].values
        lat = df_1hz['Latitude'].values
        xs, ys = _proj_utm52(lon, lat)
        ref_alt = df_1hz['Altitude'].iat[0]

        df_1hz['E'] = xs - xs[0]
        df_1hz['N'] = ys - ys[0]
        df_1hz['U'] = df_1hz['Altitude'] - ref_alt

        df_1hz['Delta X'] = df_1hz['E'].diff().fillna(0)
        df_1hz['Delta Y'] = df_1hz['N'].diff().fillna(0)
        df_1hz['Delta U'] = df_1hz['U'].diff().fillna(0)

        # --- 4) df_1hz → 50Hz 선형 보간 ---
        t_start = df_1hz['Elapsed Time'].iat[0]
        t_end   = df_1hz['Elapsed Time'].iat[-1]
        total_seconds = t_end - t_start
        num_samples  = int(total_seconds * 50) + 1

        t_interp = np.linspace(
            t_start,
            t_end,
            num=num_samples,
            endpoint=True
        )

        E_interp = np.interp(t_interp,
                             df_1hz['Elapsed Time'].values,
                             df_1hz['E'].values)
        N_interp = np.interp(t_interp,
                             df_1hz['Elapsed Time'].values,
                             df_1hz['N'].values)
        U_interp = np.interp(t_interp,
                             df_1hz['Elapsed Time'].values,
                             df_1hz['U'].values)

        df_50hz = pd.DataFrame({
            'Elapsed Time': t_interp,
            'E': E_interp,
            'N': N_interp,
            'U': U_interp
        })
        df_50hz['Delta X'] = df_50hz['E'].diff().fillna(0)
        df_50hz['Delta Y'] = df_50hz['N'].diff().fillna(0)
        df_50hz['Delta U'] = df_50hz['U'].diff().fillna(0)
        
        df_all = pd.concat([df, df_50hz[['E','N']]], axis=1)
        df_all = df_all.dropna(subset=['E','N']).reset_index(drop=True)
        
        # window_size 설정
        window_size = 50
        lag = window_size - 1  # 49

        # 1) i → i+window_size 간격의 ΔE, ΔN 계산 (shift 활용)
        df_all['delta_E'] = df_all['E'].shift(-window_size) - df_all['E']
        df_all['delta_N'] = df_all['N'].shift(-window_size) - df_all['N']

        # 2) 거리 변화량(dist_change)과 방향 변화량(heading_change) 계산
        df_all['dist_change']    = np.hypot(df_all['delta_E'], df_all['delta_N'])
        df_all['heading_change'] = np.arctan2(df_all['delta_N'], df_all['delta_E'])

        # 절대 방위의 차분
        df_all['heading_diff'] = df_all['heading_change'] - df_all['heading_change'].shift(50)
        df_all['heading_diff'] = ((df_all['heading_diff'] + np.pi) % (2*np.pi)) - np.pi
        df_all['heading_diff_deg'] = np.degrees(df_all['heading_diff'])
        
        sensor_cols = [
            'Accelerometer x', 'Accelerometer y', 'Accelerometer z',
            'Gyroscope x', 'Gyroscope y', 'Gyroscope z',
            'Acc_Norm', 'Gyro_Norm'
        ]

        # Y로 사용할 컬럼
        y_cols = ['dist_change', 'heading_diff']

        window_size = 50
        stride = 1

        X_list = []
        Y_list = []

        n_rows = len(df_all)
        for i in range(0, n_rows - window_size, stride):
            y = df_all.loc[i + window_size, y_cols].values.astype(float)
            if pd.isnull(y).any():
                break

            x = df_all.loc[i : i + window_size - 1, sensor_cols].values.astype(float)

            X_list.append(x)
            Y_list.append(y)

        X = np.stack(X_list, axis=0)
        Y = np.stack(Y_list, axis=0)
        
        return df_all, X, Y

    @staticmethod
    def load_and_preprocess_csv_test(file_path, delimiter=',', header=0, skiprows=50):
        """
        테스트용 CSV 로드 및 간단 전처리
        """
        df = pd.read_csv(file_path, delimiter=delimiter, header=header, skiprows=skiprows)
        if len(df) > 100:
            df = df.iloc[:-100].reset_index(drop=True)

        df.columns = [
            'Time',
            'Accelerometer x', 'Accelerometer y', 'Accelerometer z',
            'Gyroscope x', 'Gyroscope y', 'Gyroscope z',
            'Magnetometer x', 'Magnetometer y', 'Magnetometer z',
            'Orientation x', 'Orientation y', 'Orientation z',
            'Pressure', 'Latitude', 'Longitude', 'Altitude', 'Speed_GPS'
        ]
        df['Time'] = pd.to_datetime(df['Time'], format='%Y-%m-%d %H:%M:%S.%f')
        start_dt = df['Time'].iloc[0]
        df['Elapsed Time'] = (df['Time'] - start_dt).dt.total_seconds()

        df['Acc_Norm']  = np.linalg.norm(df[['Accelerometer x','Accelerometer y','Accelerometer z']], axis=1)
        df['Gyro_Norm'] = np.linalg.norm(df[['Gyroscope x','Gyroscope y','Gyroscope z']], axis=1)
        
        return df
