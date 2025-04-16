import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
import logging
from datetime import datetime
from pyproj import CRS, Transformer
from scipy.interpolate import interp1d, UnivariateSpline
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
        logger.info("속도 및 헤딩 계산 시작")
        df['Speed'] = np.sqrt(df['Delta X']**2 + df['Delta Y']**2) / time_interval
        
        headings = np.arctan2(df['Delta Y'], df['Delta X']).values
        df['Heading'] = headings
        target_heading = df[df['Heading'] != 0]['Heading']
        heading_change_filtered = np.concatenate(([0], np.diff(target_heading)))
        heading_change_series = pd.Series(heading_change_filtered, index=target_heading.index)
        df['Heading Change'] = heading_change_series
        logger.info("속도 및 헤딩 계산 완료")
        return df

    @staticmethod
    def make_XY_using_dataframe(df, window_size=50, stride=1):
        """
        슬라이딩 윈도우 방식으로 X, Y 데이터 생성  
        X: 센서 데이터, Y: [속도, 헤딩 변화량]  
        ★ 센서 데이터에 Orientation 관련 cos/sin 피처를 추가함.
        """
        start_time = time.time()
        logger.info(f"X, Y 데이터 생성 시작 (window_size: {window_size}, stride: {stride})")
        
        # 헤딩 변화량이 있는 데이터 선택
        logger.info("헤딩 변화량 데이터 선택")
        target = df[df['Heading Change'].notna()][['Speed', 'Heading Change']]
        onesec_speed = target['Speed'].values
        onesec_heading_change = np.unwrap(target['Heading Change'].values)
                
        # 보간 처리
        logger.info("속도 및 헤딩 변화량 보간 시작")

        # 원본 타임 인덱스
        original_time = np.arange(len(onesec_speed))
        interp_time = np.linspace(0, len(onesec_speed) - 1, num=(len(onesec_speed) - 1) * 50 + 1)

        # 스무딩 스플라인 보간 함수 정의 (s: 스무딩 정도)
        smoothing_factor_speed = 0.5   # 값이 클수록 부드러워짐
        smoothing_factor_heading = 0.2

        speed_spline = UnivariateSpline(original_time, onesec_speed, s=smoothing_factor_speed)
        heading_spline = UnivariateSpline(original_time, onesec_heading_change, s=smoothing_factor_heading)

        # 보간 결과 생성
        interpolated_speed = speed_spline(interp_time)
        interpolated_heading_change = heading_spline(interp_time)
        
        Y_full = np.stack((interpolated_speed, interpolated_heading_change), axis=1)
        logger.info("보간 완료")
        
        # ★ 센서 데이터에 Orientation cos/sin 피처도 포함 (총 7 + 6 = 13 피처)
        sensor_columns = ['Accelerometer x', 'Accelerometer y', 'Accelerometer z',
                          'Gyroscope x', 'Gyroscope y', 'Gyroscope z',
                          'Acc_Norm',
                          'cos_roll', 'sin_roll', 'cos_pitch', 'sin_pitch', 'cos_yaw', 'sin_yaw']
        
        sensor_data = df[sensor_columns].values
        logger.info("센서 데이터 추출")
        
        min_length = min(len(sensor_data), len(Y_full))
        sensor_data = sensor_data[:min_length]
        Y_full = Y_full[:min_length]
        
        logger.info("슬라이딩 윈도우 적용")
        X_list = []
        Y_list = []
        for i in range(0, min_length - window_size + 1, stride):
            X_window = sensor_data[i:i + window_size]
            Y_target = Y_full[i + window_size - 1]
            X_list.append(X_window)
            Y_list.append(Y_target)
            
        X = np.array(X_list)
        Y = np.array(Y_list)
        
        end_time = time.time()
        logger.info(f"X, Y 데이터 생성 완료: {end_time - start_time:.2f} 초 소요")
        logger.info(f"생성된 데이터 크기 - X: {X.shape}, Y: {Y.shape}")
        return X, Y

    @staticmethod
    def plot_label_distribution(Y, save_path=None):
        """
        라벨 분포를 10도 단위로 그룹화하여 시각화
        """
        import os
        import matplotlib.pyplot as plt
        import numpy as np
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
        # 첫 번째 label: 속도, 두 번째 label: heading change (라디안 단위 → 도 단위 변환)
        speeds = Y[:, 0]
        heading_changes = np.rad2deg(Y[:, 1])
        
        # 10도 단위로 그룹화
        heading_bins = np.arange(-180, 181, 10)
        speed_bins = np.arange(0, np.max(speeds) + 0.1, 0.1)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 속도 분포
        speed_counts, speed_edges = np.histogram(speeds, bins=speed_bins)
        speed_probs = speed_counts / len(speeds)
        ax1.bar(speed_edges[:-1], speed_probs, width=np.diff(speed_edges), align='edge')
        ax1.set_title('Speed Distribution')
        ax1.set_xlabel('Speed (m/s)')
        ax1.set_ylabel('Probability')
        ax1.grid(True, alpha=0.3)
        
        # 헤딩 변화량 분포
        heading_counts, heading_edges = np.histogram(heading_changes, bins=heading_bins)
        heading_probs = heading_counts / len(heading_changes)
        ax2.bar(heading_edges[:-1], heading_probs, width=np.diff(heading_edges), align='edge')
        ax2.set_title('Heading Change Distribution (10° bins)')
        ax2.set_xlabel('Heading Change (degrees)')
        ax2.set_ylabel('Probability')
        ax2.grid(True, alpha=0.3)
        
        # 각 그룹의 확률 출력
        print("\n=== Speed Distribution ===")
        for i in range(len(speed_probs)):
            if speed_probs[i] > 0:
                print(f"Speed {speed_edges[i]:.1f}-{speed_edges[i+1]:.1f} m/s: {speed_probs[i]:.4f}")
        
        print("\n=== Heading Change Distribution ===")
        for i in range(len(heading_probs)):
            if heading_probs[i] > 0:
                print(f"Heading {heading_edges[i]:.0f}-{heading_edges[i+1]:.0f}°: {heading_probs[i]:.4f}")
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved: {save_path}")
        plt.show()
