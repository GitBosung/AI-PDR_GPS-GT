# src/data_processor.py
import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
import logging
from datetime import datetime
from pyproj import CRS, Transformer
from scipy.interpolate import interp1d

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
        
        end_time = time.time()
        logger.info("전체 전처리 완료")
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
        original_time = np.arange(len(onesec_speed))
        interp_time = np.linspace(0, len(onesec_speed) - 1, num=(len(onesec_speed) - 1) * 50 + 1)
        speed_interp_func = interp1d(original_time, onesec_speed, kind='linear')
        heading_interp_func = interp1d(original_time, onesec_heading_change, kind='linear')
        interpolated_speed = speed_interp_func(interp_time)
        interpolated_heading_change = heading_interp_func(interp_time)
        
        Y_full = np.stack((interpolated_speed, interpolated_heading_change), axis=1)
        logger.info("보간 완료")
        
        sensor_columns = ['Accelerometer x', 'Accelerometer y', 'Accelerometer z',
                          'Gyroscope x', 'Gyroscope y', 'Gyroscope z',
                          'Acc_Norm']
        
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

    def plot_label_distribution(Y, save_path=None):
        import os
        import matplotlib.pyplot as plt
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
        # 첫 번째 label: 속도, 두 번째 label: heading change (라디안 단위 → 도 단위 변환)
        speeds = Y[:, 0]
        heading_changes = np.rad2deg(Y[:, 1])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # density=True 옵션을 통해 확률 밀도 분포로 변경
        ax1.hist(speeds, bins=50, density=True, color='blue', alpha=0.7)
        ax1.set_title('Speed Probability Distribution')
        ax1.set_xlabel('Speed (m/s)')
        ax1.set_ylabel('Probability')
        
        ax2.hist(heading_changes, bins=50, density=True, color='red', alpha=0.7)
        ax2.set_title('Heading Change Probability Distribution')
        ax2.set_xlabel('Heading Change (degrees)')
        ax2.set_ylabel('Probability')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved: {save_path}")  # logger 대신 print 사용 (또는 logger 사용)
        plt.show()
        print("Target label distribution visualization completed")

