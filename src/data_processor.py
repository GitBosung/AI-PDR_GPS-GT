import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
import logging
from datetime import datetime
from pyproj import CRS, Transformer
from scipy.interpolate import PchipInterpolator
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
        logger.info(f"CSV 파일 로딩 완료: {len(df)} 행")
        
        # 컬럼 이름 재정의
        new_columns = [
            'Time', 
            'Accelerometer x', 'Accelerometer y', 'Accelerometer z',
            'Gyroscope x', 'Gyroscope y', 'Gyroscope z',
            'Magnetometer x', 'Magnetometer y', 'Magnetometer z',
            'Orientation x', 'Orientation y', 'Orientation z',
            'Pressure', 'Latitude', 'Longitude', 'Altitude', 'Speed_GPS'
        ]
        df.columns = new_columns
        logger.info("컬럼 이름 재정의 완료")
        
        # 시간 전처리
        logger.info("시간 데이터 전처리 시작")
        df['Time'] = pd.to_datetime(df['Time'], format='%Y-%m-%d %H:%M:%S.%f')
        start_time_dt = df['Time'].iloc[0]
        df['Elapsed Time'] = (df['Time'] - start_time_dt).dt.total_seconds()
        logger.info("시간 데이터 전처리 완료")
        
        # GPS 데이터 전처리 (벡터화)
        logger.info("GPS 데이터 전처리 시작")
        df = DataProcessor.process_gps_data(df)
        logger.info("GPS 데이터 전처리 완료")
        
        # 가속도 벡터의 크기 계산
        logger.info("가속도 벡터 크기 계산 시작")
        df['Acc_Norm'] = np.sqrt(
            df['Accelerometer x']**2 +
            df['Accelerometer y']**2 +
            df['Accelerometer z']**2
        )
        logger.info("가속도 벡터 크기 계산 완료")
        
        # 속도 및 헤딩 계산
        logger.info("속도 및 헤딩 계산 시작")
        df = DataProcessor.compute_speed_and_heading(df)
        logger.info("속도 및 헤딩 계산 완료")
        
        # Orientation → Euler → cos/sin 값
        logger.info("Orientation 데이터 변환 시작")
        df = DataProcessor.compute_orientation_features(df)
        logger.info("Orientation 데이터 변환 완료")
        
        end_time = time.time()
        logger.info(f"전체 전처리 완료 (소요 {end_time - start_time:.2f}s)")
        return df

    @staticmethod
    def process_gps_data(df):
        """
        ENU 변환을 배열 연산으로 처리하여 속도를 개선.
        """
        logger.info("ENU 좌표 변환 시작")
        # 기준점
        ref_lat = df['Latitude'].iat[0]
        ref_lon = df['Longitude'].iat[0]
        ref_alt = df['Altitude'].iat[0]

        # Transformer 한 번만 생성
        wgs84  = CRS.from_epsg(4326)
        utm52  = CRS.from_epsg(32652)
        transformer = Transformer.from_crs(wgs84, utm52, always_xy=True)

        # 배열 단위 변환 (lon, lat 순서)
        lons = df['Longitude'].values
        lats = df['Latitude'].values
        alts = df['Altitude'].values

        xs, ys = transformer.transform(lons, lats)  # shape (N,)
        # 기준 좌표
        ref_x, ref_y = xs[0], ys[0]

        # E, N, U 계산
        df['E'] = xs - ref_x
        df['N'] = ys - ref_y
        df['U'] = alts - ref_alt

        # 차분
        df['Delta X'] = df['E'].diff().fillna(0)
        df['Delta Y'] = df['N'].diff().fillna(0)
        logger.info("ENU 좌표 변환 완료")
        return df

    @staticmethod
    def compute_speed_and_heading(df, time_interval=1.0):
        # ΔX, ΔY, ΔD, Speed
        df['Delta X'] = df['E'].diff().fillna(0)
        df['Delta Y'] = df['N'].diff().fillna(0)
        df['Delta D'] = np.hypot(df['Delta X'], df['Delta Y'])
        df['Speed'] = df['Delta D']

        # 이동 시점만 필터
        move_mask = df['Delta D'] > 0
        df_move = df[move_mask].copy()

        # Heading 계산 및 wrap 보정
        df_move['Heading'] = np.arctan2(df_move['Delta Y'], df_move['Delta X'])
        df_move['Heading'] = df_move['Heading'].fillna(method='ffill').fillna(0)
        hc = df_move['Heading'].diff().fillna(0)
        df_move['Pure Heading Change'] = ((hc + np.pi) % (2 * np.pi)) - np.pi

        # 원본 매핑
        df['Pure Speed'] = 0.0
        df['Pure Heading Change'] = 0.0
        df.loc[df_move.index, 'Pure Speed'] = df_move['Speed']
        df.loc[df_move.index, 'Pure Heading Change'] = df_move['Pure Heading Change']
        df['Pure Heading Change (deg)'] = np.degrees(df['Pure Heading Change'])

        return df

    @staticmethod
    def compute_orientation_features(df):
        """
        Orientation x,y,z 로부터 단위 쿼터니언 [w,x,y,z] 복원하고,
        회전행렬 R을 구해 첫 두 열을 6D feature로 추가합니다.
        """
        ox = df['Orientation x'].values
        oy = df['Orientation y'].values
        oz = df['Orientation z'].values
        # 쿼터니언 w 복원
        w  = np.sqrt(np.maximum(0, 1 - (ox**2 + oy**2 + oz**2)))
        # 쿼터니언 배열 (N,4)
        quat = np.stack([w, ox, oy, oz], axis=1)

        # quaternion -> rotation matrix R (N,3,3)
        # 벡터화된 공식
        x, y, z = quat[:,1], quat[:,2], quat[:,3]
        w = quat[:,0]
        R00 = 1 - 2*(y*y + z*z)
        R01 = 2*(x*y - z*w)
        R02 = 2*(x*z + y*w)
        R10 = 2*(x*y + z*w)
        R11 = 1 - 2*(x*x + z*z)
        R12 = 2*(y*z - x*w)
        R20 = 2*(x*z - y*w)
        R21 = 2*(y*z + x*w)
        R22 = 1 - 2*(x*x + y*y)

        # (N,3,3) 으로 reshape
        R = np.stack([
            np.stack([R00, R01, R02], axis=1),
            np.stack([R10, R11, R12], axis=1),
            np.stack([R20, R21, R22], axis=1),
        ], axis=1)

        # 앞 두 열만 꺼내 6D feature로
        r1 = R[:, :, 0]             # (N,3)
        r2 = R[:, :, 1]             # (N,3)
        rot6 = np.concatenate([r1, r2], axis=1)  # (N,6)

        # DataFrame에 컬럼 추가
        for i in range(6):
            df[f'rot6_{i}'] = rot6[:, i]

        return df

    @staticmethod
    def make_XY_using_dataframe(df, window_size=50, stride=1, freq=50):
        """
        슬라이딩 윈도우 방식으로 X, Y 데이터 생성
        """
        from numpy.lib.stride_tricks import sliding_window_view

        logger.info(
            f"X, Y 생성 시작 (window_size={window_size}, stride={stride}, freq={freq})"
        )
        # 보간
        t_full = df['Elapsed Time'].values
        t_spd  = df.loc[df['Pure Speed'] != 0, 'Elapsed Time'].values
        v_spd  = df.loc[df['Pure Speed'] != 0, 'Pure Speed'].values
        t_hc   = df.loc[df['Pure Heading Change'] != 0, 'Elapsed Time'].values
        hc     = df.loc[df['Pure Heading Change'] != 0, 'Pure Heading Change'].values

        spd_pchip = PchipInterpolator(t_spd, v_spd, extrapolate=False)
        hc_pchip  = PchipInterpolator(t_hc, hc,   extrapolate=False)
        v_full = np.nan_to_num(spd_pchip(t_full), nan=0.0)
        hc_full = np.nan_to_num(hc_pchip(t_full), nan=0.0)
        Y_full = np.stack((v_full, hc_full), axis=1)

        # 센서 데이터
        sensor_columns = [
            'Accelerometer x','Accelerometer y','Accelerometer z',
            'Gyroscope x','Gyroscope y','Gyroscope z',
            'Acc_Norm',
            # 6D 회전 피처 추가
            'rot6_0','rot6_1','rot6_2','rot6_3','rot6_4','rot6_5'
        ]
        
        sensor_data = df[sensor_columns].values

        min_len = min(len(sensor_data), len(Y_full))
        sensor_data = sensor_data[:min_len]
        Y_full      = Y_full[:min_len]

        # 슬라이딩 윈도우 벡터화
        windows = sliding_window_view(sensor_data, window_shape=(window_size, sensor_data.shape[1]))
        windows = windows.squeeze(1)          # (T-window+1, window_size, features)
        X = windows[::stride]
        Y = Y_full[window_size - 1 :: stride]

        X = X.astype(np.float32)
        Y = Y.astype(np.float32)
        logger.info(f"X, Y 생성 완료: X={X.shape}, Y={Y.shape}")
        return X, Y
