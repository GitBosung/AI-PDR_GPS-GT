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
    def make_XY_using_dataframe(df, window_size=50, stride=1):
        """
        df: 'Pure Speed', 'Pure Heading Change' 칼럼이 이미 들어있는 DataFrame
        window_size:  윈도우 크기 (샘플)
        stride:       X 윈도우를 자를 스트라이드
        freq:         센서 샘플링 주파수 (Hz)
        guard:        윈도우 앞뒤로 볼 가드 영역 (샘플)
        eps:          헤딩 변화 최소 감지 임계치 (rad)
        """
        import numpy as np
        from numpy.lib.stride_tricks import sliding_window_view

        # 전체 시간축
        t_full = df['Elapsed Time'].values  # (T,)

        # 1) 1초(비중첩) 윈도우 단위 레이블링
        speed = df['Pure Speed'].values               # (T,)
        hc    = df['Pure Heading Change'].values      # (T,)
        
        real_speed = []
        real_heading = []
        
        for i in range(50, len(speed), 50):
            max_speed = 0
            max_heading = 0
            for j in range(i-5, i+5):
                if speed[j] > max_speed:
                    max_speed = speed[j]
                if abs(hc[j]) > max_heading:
                    max_heading = abs(hc[j])
            real_speed.append(max_speed)
            real_heading.append(max_heading)
            
        real_speed = np.array(real_speed)
        real_heading = np.array(real_heading)
        
        t_orig = np.arange(0, len(real_speed))
        t_new = np.arange(0, len(real_speed), 0.02)
        
        f = PchipInterpolator(t_orig, real_speed)
        real_speed_interp = f(t_new)
        
        f = PchipInterpolator(t_orig, real_heading)
        real_heading_interp = f(t_new)
        
        v_full = real_speed_interp
        hc_full = real_heading_interp
        
        Y_full = np.stack([v_full, hc_full], axis=1)  # (T,2)
        


        # 3) 슬라이딩 윈도우로 X, Y 매칭
        sensor_cols = [
            'Accelerometer x','Accelerometer y','Accelerometer z',
            'Gyroscope x','Gyroscope y','Gyroscope z',
            'Acc_Norm','Gyro_Norm',
        ]
        data = df[sensor_cols].values.astype(np.float32)  # (T,8)

        # X: (num_windows, window_size, num_features)
        windows = sliding_window_view(data, (window_size, data.shape[1])).squeeze(1)
        X = windows[::stride]
        

        n_windows = X.shape[0]
        Y = Y_full[:n_windows]  # shape = (n_windows, 2)
        
        print(len(Y))
        print(len(X))

        logger.info(f"make_XY 완료: X={X.shape}, Y={Y.shape}")
        return X, Y
