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
        new_columns = ['Time', 
                       'Accelerometer x', 'Accelerometer y', 'Accelerometer z',
                       'Gyroscope x', 'Gyroscope y', 'Gyroscope z',
                       'Magnetometer x', 'Magnetometer y', 'Magnetometer z',
                       'Orientation x', 'Orientation y', 'Orientation z',
                       'Pressure', 'Latitude', 'Longitude', 'Altitude', 'Speed_GPS']
        df.columns = new_columns
        logger.info("컬럼 이름 재정의 완료")
        
        # 시간 전처리
        logger.info("시간 데이터 전처리 시작")
        df['Time'] = pd.to_datetime(df['Time'], format='%Y-%m-%d %H:%M:%S.%f')
        start_time_dt = df['Time'].iloc[0]
        df['Elapsed Time'] = (df['Time'] - start_time_dt).dt.total_seconds()
        logger.info("시간 데이터 전처리 완료")
        
        # GPS 데이터 전처리
        logger.info("GPS 데이터 전처리 시작")
        df = DataProcessor.process_gps_data(df)
        logger.info("GPS 데이터 전처리 완료")
        
        # 가속도 벡터의 크기 계산
        logger.info("가속도 벡터 크기 계산 시작")
        df['Acc_Norm'] = np.sqrt(df['Accelerometer x']**2 + 
                                 df['Accelerometer y']**2 + 
                                 df['Accelerometer z']**2)
        logger.info("가속도 벡터 크기 계산 완료")
        
        # 속도 및 헤딩 계산
        logger.info("속도 및 헤딩 계산 시작")
        df = DataProcessor.compute_speed_and_heading(df)
        logger.info("속도 및 헤딩 계산 완료")
        
        # ★ Orientation (쿼터니언) → Euler (roll, pitch, yaw) 변환 후 cos, sin값 계산 ★
        logger.info("Orientation 데이터 변환 (쿼터니언 → Euler 각, cos/sin값) 시작")
        df = DataProcessor.compute_orientation_features(df)
        logger.info("Orientation 데이터 변환 완료")
        
        end_time = time.time()
        logger.info("전체 전처리 완료")
        return df

    @staticmethod
    def compute_orientation_features(df):
        """
        쿼터니언의 벡터 성분 (Orientation x, y, z)을 사용하여
        w를 계산한 후, Euler 각 (roll, pitch, yaw)를 구하고,
        각 각도의 cos 및 sin 값을 계산하여 새로운 피처로 추가.
        스케일러는 적용하지 않음.
        """
        ox = df['Orientation x'].values
        oy = df['Orientation y'].values
        oz = df['Orientation z'].values
        
        # 쿼터니언이 정규화되어 있다고 가정하고, w 계산 (음수가 되지 않도록 처리)
        w = np.sqrt(np.maximum(0, 1 - (ox**2 + oy**2 + oz**2)))
        
        # Euler 각 계산 (roll, pitch, yaw) – 아래 공식은 일반적인 변환 공식입니다.
        roll = np.arctan2(2*(w*ox + oy*oz), 1 - 2*(ox**2 + oy**2))
        pitch = np.arcsin(2*(w*oy - oz*ox))
        yaw = np.arctan2(2*(w*oz + ox*oy), 1 - 2*(oy**2 + oz**2))
        
        # cos, sin 값 계산
        df['cos_roll'] = np.cos(roll)
        df['sin_roll'] = np.sin(roll)
        df['cos_pitch'] = np.cos(pitch)
        df['sin_pitch'] = np.sin(pitch)
        df['cos_yaw'] = np.cos(yaw)
        df['sin_yaw'] = np.sin(yaw)
        
        return df

    @staticmethod
    def llh_to_enu(lat, lon, alt, ref_lat, ref_lon, ref_alt):
        wgs84 = CRS.from_epsg(4326)
        utm52 = CRS.from_epsg(32652)
        transformer = Transformer.from_crs(wgs84, utm52, always_xy=True)
        x, y = transformer.transform(lon, lat)
        ref_x, ref_y = transformer.transform(ref_lon, ref_lat)
        e = x - ref_x
        n = y - ref_y
        u = alt - ref_alt
        return e, n, u

    @staticmethod
    def process_gps_data(df):
        logger.info("ENU 좌표 변환 시작")
        ref_lat = df['Latitude'].iloc[0]
        ref_lon = df['Longitude'].iloc[0]
        ref_alt = df['Altitude'].iloc[0]
        enu_coords = np.array([
            DataProcessor.llh_to_enu(lat, lon, alt, ref_lat, ref_lon, ref_alt) 
            for lat, lon, alt in zip(df['Latitude'], df['Longitude'], df['Altitude'])
        ])
        df['E'] = enu_coords[:, 0]
        df['N'] = enu_coords[:, 1]
        df['U'] = enu_coords[:, 2]
        df['Delta X'] = df['E'].diff().fillna(0)
        df['Delta Y'] = df['N'].diff().fillna(0)
        logger.info("ENU 좌표 변환 완료")
        return df

    @staticmethod
    def compute_speed_and_heading(df, time_interval=1.0):
        """
        1) ΔX, ΔY, ΔD 계산
        2) GPS가 이동한 시점(move_mask)만 골라 df_move 생성
        3) df_move 에서 절대 Heading → diff → 순수 Heading Change 계산
        4) 원본 df 에 Pure Speed, Pure Heading Change 매핑
        """
        # 1) ΔX, ΔY, ΔD, Speed 계산
        df['Delta X'] = df['E'].diff().fillna(0)
        df['Delta Y'] = df['N'].diff().fillna(0)
        df['Delta D'] = np.hypot(df['Delta X'], df['Delta Y'])
        df['Speed']   = df['Delta D'] 

        # 2) GPS가 실제 이동한 인덱스만
        move_mask = df['Delta D'] > 0
        df_move   = df[move_mask].copy()

        # 3) df_move 에서 절대 Heading (라디안)
        df_move['Heading'] = np.arctan2(df_move['Delta Y'], df_move['Delta X'])

        #    → 일시적인 NaN 제거
        df_move['Heading'] = df_move['Heading'].fillna(method='ffill').fillna(0)

        #    Heading Change 계산 & wrap 보정
        hc = df_move['Heading'].diff().fillna(0)
        df_move['Pure Heading Change'] = ((hc + np.pi) % (2 * np.pi)) - np.pi

        # 4) 원본 df 에 매핑
        df['Pure Speed'] = 0.0
        df['Pure Heading Change'] = 0.0

        # 이동 시점 인덱스에는 실제 Speed, Heading Change 할당
        df.loc[df_move.index, 'Pure Speed'] = df_move['Speed']
        df.loc[df_move.index, 'Pure Heading Change'] = df_move['Pure Heading Change']

        # (선택) 도 단위 컬럼
        df['Pure Heading Change (deg)'] = np.degrees(df['Pure Heading Change'])

        return df

    @staticmethod
    def make_XY_using_dataframe(df, window_size=50, stride=1, freq=50):
        """
        슬라이딩 윈도우 방식으로 X, Y 데이터 생성  
        X: 센서 데이터 (Acc, Gyro, Acc_Norm, cos/sin Orientation)  
        Y: [Speed_50Hz, HC_50Hz]
        """
        logger.info(f"X, Y 데이터 생성 시작 (window_size={window_size}, stride={stride}, freq={freq})")
        
        # 1) 전체 시간축 및 이벤트 시점만 추출
        t_full = df['Elapsed Time'].values
        t_spd = df.loc[df['Pure Speed'] != 0, 'Elapsed Time'].values
        v_spd = df.loc[df['Pure Speed'] != 0, 'Pure Speed'].values
        t_hc  = df.loc[df['Pure Heading Change'] != 0, 'Elapsed Time'].values
        hc    = df.loc[df['Pure Heading Change'] != 0, 'Pure Heading Change'].values

        # 2) PCHIP 보간기 생성 (경계 외삽은 NaN)

        spd_pchip = PchipInterpolator(t_spd, v_spd, extrapolate=False)
        hc_pchip  = PchipInterpolator(t_hc, hc, extrapolate=False)

        # 3) 전체 프레임에 보간 적용 & NaN을 0으로
        v_full = spd_pchip(t_full)
        hc_full = hc_pchip(t_full)
        v_full = np.nan_to_num(v_full, nan=0.0)
        hc_full = np.nan_to_num(hc_full, nan=0.0)

        # 4) Y_full 배열 생성
        Y_full = np.stack((v_full, hc_full), axis=1)
        logger.info("보간 완료: Y_full shape = %s", Y_full.shape)

        # 5) 센서 피처 추출
        sensor_columns = [
            'Accelerometer x','Accelerometer y','Accelerometer z',
            'Gyroscope x','Gyroscope y','Gyroscope z',
            'Acc_Norm']
        sensor_data = df[sensor_columns].values

        # 6) 길이 맞추기
        min_len = min(len(sensor_data), len(Y_full))
        sensor_data = sensor_data[:min_len]
        Y_full = Y_full[:min_len]

        # 7) 슬라이딩 윈도우 생성
        X_list, Y_list = [], []
        for i in range(0, min_len - window_size + 1, stride):
            X_list.append(sensor_data[i:i+window_size])
            Y_list.append(Y_full[i+window_size-1])

        X = np.array(X_list, dtype=np.float32)
        Y = np.array(Y_list, dtype=np.float32)

        logger.info("X, Y 데이터 생성 완료: X=%s, Y=%s", X.shape, Y.shape)
        return X, Y

