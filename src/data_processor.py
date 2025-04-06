import numpy as np
import pandas as pd
from datetime import datetime
from pyproj import CRS, Transformer
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import logging
import time
import seaborn as sns
import os

# 전역 로거 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class DataProcessor:
    """
    데이터 로딩 및 전처리를 수행하는 클래스.
    
    주요 기능:
    - CSV 파일 로딩 및 초기 전처리
    - GPS 데이터(위경도)를 ENU 좌표로 변환
    - 가속도 및 헤딩 등의 추가 피처 계산
    - 슬라이딩 윈도우 기반의 X, Y 데이터 생성
    """
    def __init__(self):
        pass

    @staticmethod
    def load_and_preprocess_csv(file_path, delimiter=',', header=0, skiprows=300):
        """
        CSV 파일을 로드하고 전처리를 수행합니다.
        
        매개변수:
          - file_path: CSV 파일 경로
          - delimiter: 구분자 (기본값: ',')
          - header: 헤더 행 (기본값: 0)
          - skiprows: 처음 몇 줄을 건너뛸지 (기본값: 300)
        
        처리 내용:
          1. 컬럼명 재정의 및 시간 데이터 처리
          2. GPS 데이터를 ENU 좌표로 변환하고, Delta 값을 계산
          3. 가속도 벡터 크기 계산
          4. 속도 및 헤딩, 헤딩 변화량 계산
        """
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
        start_time = df['Time'].iloc[0]
        df['Elapsed Time'] = (df['Time'] - start_time).dt.total_seconds()
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
        logger.info(f"전체 전처리 완료")
        
        return df

    @staticmethod
    def llh_to_enu(lat, lon, alt, ref_lat, ref_lon, ref_alt):
        """
        위도, 경도, 고도를 ENU 좌표로 변환합니다.
        
        매개변수:
          - lat, lon, alt: 변환할 위도, 경도, 고도
          - ref_lat, ref_lon, ref_alt: 기준 좌표
          
        반환값:
          - e, n, u: 각각 동(East), 북(North), 상(Ups) 좌표값
        """
        wgs84 = CRS.from_epsg(4326)       # WGS84 좌표계
        utm52 = CRS.from_epsg(32652)        # UTM 52N (한국 기준)
        transformer = Transformer.from_crs(wgs84, utm52, always_xy=True)
        
        x, y = transformer.transform(lon, lat)
        ref_x, ref_y = transformer.transform(ref_lon, ref_lat)
        
        e = x - ref_x
        n = y - ref_y
        u = alt - ref_alt
        return e, n, u

    @staticmethod
    def process_gps_data(df):
        """
        GPS 데이터를 ENU 좌표로 변환하고, delta 값(변화량)을 계산합니다.
        
        매개변수:
          - df: 데이터프레임
          
        반환값:
          - ENU 좌표 및 Delta 값이 추가된 데이터프레임
        """
        logger.info("ENU 좌표 변환 시작")
        
        # 기준 좌표: 첫 번째 행의 위도, 경도, 고도
        ref_lat = df['Latitude'].iloc[0]
        ref_lon = df['Longitude'].iloc[0]
        ref_alt = df['Altitude'].iloc[0]
        
        # 각 행의 ENU 좌표 계산
        enu_coords = np.array([
            DataProcessor.llh_to_enu(lat, lon, alt, ref_lat, ref_lon, ref_alt) 
            for lat, lon, alt in zip(df['Latitude'], df['Longitude'], df['Altitude'])
        ])
        df['E'] = enu_coords[:, 0]
        df['N'] = enu_coords[:, 1]
        df['U'] = enu_coords[:, 2]
        
        # Delta 값 계산
        df['Delta X'] = df['E'].diff().fillna(0)
        df['Delta Y'] = df['N'].diff().fillna(0)
        
        logger.info("ENU 좌표 변환 완료")
        return df

    @staticmethod
    def compute_speed_and_heading(df, time_interval=1.0):
        """
        센서의 Delta 값을 바탕으로 속도와 헤딩, 헤딩 변화량을 계산합니다.
        
        매개변수:
          - df: 데이터프레임
          - time_interval: 시간 간격 (기본값: 1초)
          
        반환값:
          - 속도, 헤딩, 헤딩 변화량이 추가된 데이터프레임
        """
        logger.info("속도 및 헤딩 계산 시작")
        
        # 속도 계산
        df['Speed'] = np.sqrt(df['Delta X']**2 + df['Delta Y']**2) / time_interval
        
        # 각 시점의 헤딩(라디안) 계산
        headings = np.arctan2(df['Delta Y'], df['Delta X']).values
        df['Heading'] = headings
        
        # 헤딩 변화량 계산 (헤딩 값이 0이 아닌 데이터만 선택)
        target_heading = df[df['Heading'] != 0]['Heading']
        heading_change_filtered = np.concatenate(([0], np.diff(target_heading)))
        heading_change_series = pd.Series(heading_change_filtered, index=target_heading.index)
        df['Heading Change'] = heading_change_series
        
        logger.info("속도 및 헤딩 계산 완료")
        return df

    @staticmethod
    def make_XY_using_dataframe(df, window_size=50, stride=1):
        """
        슬라이딩 윈도우를 사용하여 모델 입력(X)와 목표값(Y)을 생성합니다.
        
        X: 센서 데이터 (Accelerometer, Gyroscope, Orientation 등)
        Y: [속도, 1초 동안의 헤딩 변화량]
        
        매개변수:
          - df: 전처리된 데이터프레임
          - window_size: 한 윈도우의 샘플 수 (기본값: 50)
          - stride: 윈도우 이동 간격 (기본값: 1)
          
        반환값:
          - X, Y numpy 배열
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
        
        # 센서 데이터 추출
        logger.info("센서 데이터 추출")
        sensor_columns = ['Accelerometer x', 'Accelerometer y', 'Accelerometer z',
                          'Gyroscope x', 'Gyroscope y', 'Gyroscope z']
        sensor_data = df[sensor_columns].values
        
        # 데이터 길이 맞추기
        logger.info("데이터 길이 맞추기")
        min_length = min(len(sensor_data), len(Y_full))
        sensor_data = sensor_data[:min_length]
        Y_full = Y_full[:min_length]
        
        # 슬라이딩 윈도우 적용
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
        정답 레이블의 분포를 시각화합니다.
        
        Args:
            Y (numpy.ndarray): 정답 레이블 배열 (속도, 헤딩 변화량)
            save_path (str, optional): 그래프를 저장할 경로
        """
        # 디렉토리가 없으면 생성
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 속도와 헤딩 변화량 분리
        speeds = Y[:, 0]
        heading_changes = np.rad2deg(Y[:, 1])  # 라디안을 도(degree)로 변환
        
        # 그래프 생성
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 속도 분포
        ax1.hist(speeds, bins=50, color='blue', alpha=0.7)
        ax1.set_title('Speed Distribution')
        ax1.set_xlabel('Speed (m/s)')
        ax1.set_ylabel('Frequency')
        
        # 헤딩 변화량 분포
        ax2.hist(heading_changes, bins=50, color='red', alpha=0.7)
        ax2.set_title('Heading Change Distribution')
        ax2.set_xlabel('Heading Change (degrees)')
        ax2.set_ylabel('Frequency')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Plot saved: {save_path}")
        
        plt.show()
        logger.info("Target label distribution visualization completed")
