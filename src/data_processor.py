import numpy as np
import pandas as pd
import os
import logging
from datetime import datetime
from pyproj import Proj
from scipy.interpolate import interp1d
import math

# 로거 설정
logger = logging.getLogger(__name__)

# -----------------------------------------------------------
# 전역(글로벌) Proj 객체: UTM zone 52N (EPSG:32652)
# 위경도 좌표를 UTM(동-북) 좌표로 변환하기 위해 사용
# -----------------------------------------------------------
_proj_utm52 = Proj("epsg:32652")


class DataProcessor:
    """
    데이터 로딩 및 전처리, 슬라이딩 윈도우 기반 X/Y 생성 기능을 제공합니다.
    주요 기능:
      - CSV 파일 읽기
      - 시간 보간 및 센서 노름 계산
      - 위경도 → ENU 좌표 변환
      - 1Hz 기준 궤적 계산 및 보간
      - 50Hz 센서 윈도우 생성
      - 슬라이딩 윈도우 X, Y 데이터 반환
    """

    def __init__(self):
        # 인스턴스 변수 없음
        pass

    @staticmethod
    def load_and_preprocess_csv(path, skiprows=300):
        """
        Parameters:
            path (str): CSV 파일 경로
            skiprows (int): 헤더 외 건너뛸 행 수 (default=300)

        Returns:
            df (pd.DataFrame): 전처리된 원본 DataFrame (속도/헤딩 컬럼 추가됨)
            X  (np.ndarray) : shape = (usable_windows, 50, 8)
            Y  (np.ndarray) : shape = (usable_windows, 2) → [speed_1, heading_1]
        """
        # ------------------------------------------------------------------
        # 1) CSV 파일 읽기 및 열 이름 지정
        # ------------------------------------------------------------------
        df = pd.read_csv(path, skiprows=skiprows)
        df.columns = [
            'Time',
            'Accelerometer x', 'Accelerometer y', 'Accelerometer z',
            'Gyroscope x', 'Gyroscope y', 'Gyroscope z',
            'Magnetometer x', 'Magnetometer y', 'Magnetometer z',
            'Orientation x', 'Orientation y', 'Orientation z',
            'Pressure', 'Latitude', 'Longitude', 'Altitude', 'Speed_GPS'
        ]

        # ------------------------------------------------------------------
        # 2) Time → datetime 변환 및 경과 시간(초) 계산
        # ------------------------------------------------------------------
        df['Time'] = pd.to_datetime(df['Time'], format='%Y-%m-%d %H:%M:%S.%f')
        start_dt = df['Time'].iloc[0]
        df['Elapsed Time'] = (df['Time'] - start_dt).dt.total_seconds()

        # ------------------------------------------------------------------
        # 3) Accelerometer, Gyroscope 노름(Norm) 계산
        #    - 센서 3축 값을 하나의 크기로 합산
        # ------------------------------------------------------------------
        df['Acc_Norm'] = np.sqrt(
            df['Accelerometer x']**2 +
            df['Accelerometer y']**2 +
            df['Accelerometer z']**2
        )
        df['Gyro_Norm'] = np.sqrt(
            df['Gyroscope x']**2 +
            df['Gyroscope y']**2 +
            df['Gyroscope z']**2
        )

        # ------------------------------------------------------------------
        # 4) 위경도 → UTM ENU 좌표(E, N) 변환
        #    - 기준점: 첫 샘플의 위경도
        # ------------------------------------------------------------------
        lat0 = df['Latitude'].iloc[0]
        lon0 = df['Longitude'].iloc[0]
        proj_enu = Proj(proj='utm', zone=52, ellps='WGS84', south=False)
        e0, n0 = proj_enu(lon0, lat0)  # 기준점 UTM 좌표

        # 전체 샘플에 대해 UTM 좌표 계산 → 상대좌표
        e_all, n_all = proj_enu(df['Longitude'].values, df['Latitude'].values)
        df['E'] = e_all - e0
        df['N'] = n_all - n0

        # ------------------------------------------------------------------
        # 5) 1Hz 대표 좌표 추출 (매 50샘플마다 중앙 인덱스)
        #    - M: 전체 50Hz 샘플 수
        #    - n_windows_Hz1: 1초당 1개 좌표 개수
        # ------------------------------------------------------------------
        M = len(df)
        n_windows_Hz1 = M // 50

        df_ori_e = []
        df_ori_n = []
        for i in range(n_windows_Hz1):
            center_idx = i * 50 + 25  # 각 50샘플 블록의 중앙 인덱스
            if center_idx < M:
                df_ori_e.append(df['E'].iloc[center_idx])
                df_ori_n.append(df['N'].iloc[center_idx])
            else:
                # 남은 샘플이 50 미만인 경우 마지막 샘플 사용
                df_ori_e.append(df['E'].iloc[-1])
                df_ori_n.append(df['N'].iloc[-1])

        e = np.array(df_ori_e)
        n = np.array(df_ori_n)

        # ------------------------------------------------------------------
        # 6) 1Hz 궤적 보정: 방향 정렬 및 상대좌표 변환
        # ------------------------------------------------------------------
        temp_dx = e[1] - e[0]
        temp_dy = n[1] - n[0]
        theta = np.arctan2(temp_dy, temp_dx)  # 초기 회전 각도
        cos_a = np.cos(theta)
        sin_a = np.sin(theta)
        R = np.array([[cos_a, -sin_a],
                      [sin_a,  cos_a]])

        # 기준점(첫 좌표)을 원점으로 이동 후 회전
        coords = np.stack((e - e[0], n - n[0]))
        rotated = R @ coords
        e_corr = rotated[0, :]
        n_corr = rotated[1, :]

        # ------------------------------------------------------------------
        # 7) 거리 변화량 및 헤딩 변화량 계산 후 차분
        # ------------------------------------------------------------------
        temp_dx = np.diff(e_corr)
        temp_dy = np.diff(n_corr)
        hz1_v = np.sqrt(temp_dx**2 + temp_dy**2)                   # 1Hz 속도
        hz1_dh = np.unwrap(np.arctan2(temp_dy, temp_dx))           # 누적 헤딩
        hz1_real_dh = np.diff(hz1_dh)                               # 1Hz 헤딩 변화량

        # ------------------------------------------------------------------
        # 8) 1Hz → 50Hz 보간 (cubic spline)
        # ------------------------------------------------------------------
        N2 = len(hz1_v)
        t_old = np.arange(len(hz1_real_dh))
        t_new = np.linspace(0, len(hz1_real_dh)-1, (len(hz1_real_dh)-1)*50 + 1)
        t_old2 = np.arange(N2)
        t_new2 = np.linspace(0, N2-1, (N2-1)*50 + 1)

        f_h_cubic = interp1d(t_old,  hz1_real_dh, kind='cubic')
        f_v_cubic = interp1d(t_old2, hz1_v,       kind='cubic')
        interp_h = f_h_cubic(t_new)   # 보간된 헤딩 변화량 (50Hz)
        interp_v = f_v_cubic(t_new2)  # 보간된 속도 (50Hz)

        speed_1 = interp_v
        heading_1 = interp_h

        # ------------------------------------------------------------------
        # 9) 센서 데이터 50샘플 슬라이딩 윈도우 생성 (8채널)
        # ------------------------------------------------------------------
        sensor_windows = []
        for i in range(0, M - 50 + 1):
            window_8ch = np.stack([
                df['Accelerometer x'].iloc[i:i+50].values,
                df['Accelerometer y'].iloc[i:i+50].values,
                df['Accelerometer z'].iloc[i:i+50].values,
                df['Gyroscope x'].iloc[i:i+50].values,
                df['Gyroscope y'].iloc[i:i+50].values,
                df['Gyroscope z'].iloc[i:i+50].values,
                df['Acc_Norm'].iloc[i:i+50].values,
                df['Gyro_Norm'].iloc[i:i+50].values
            ], axis=1)
            sensor_windows.append(window_8ch)

        # ------------------------------------------------------------------
        # 10) X, Y 배열 생성: 사용 가능한 윈도우 수 = M - 99
        # ------------------------------------------------------------------
        usable_windows = min(len(sensor_windows), len(speed_1), len(heading_1))
        X_list, Y_list = [], []
        for i in range(usable_windows):
            X_list.append(sensor_windows[i])
            Y_list.append([speed_1[i], heading_1[i]])
        X = np.stack(X_list, axis=0)
        Y = np.array(Y_list)

        # ------------------------------------------------------------------
        # 11) 원본 DataFrame에 speed_1, heading_1 컬럼 추가
        # ------------------------------------------------------------------
        df['speed_1'] = np.nan
        df['heading_1'] = np.nan
        for i in range(usable_windows):
            df.at[i, 'speed_1'] = speed_1[i]
            df.at[i, 'heading_1'] = heading_1[i]

        return df, X, Y

    @staticmethod
    def load_and_preprocess_csv_test(file_path, delimiter=',', header=0, skiprows=50):
        """
        테스트용 CSV 로드 및 간단 전처리
          - 마지막 100개 샘플 제외
          - 컬럼명 지정, Time → datetime, Elapsed Time 계산
          - Acc/Gyro Norm 계산
        """
        # ------------------------------------------------------------------
        # CSV 로드 및 마지막 100개 샘플 제외
        # ------------------------------------------------------------------
        df = pd.read_csv(file_path, delimiter=delimiter, header=header, skiprows=skiprows)
        if len(df) > 100:
            df = df.iloc[:-100].reset_index(drop=True)

        # ------------------------------------------------------------------
        # 컬럼명 지정 및 시간 정보 변환
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # Accelerometer / Gyroscope 노름 계산
        # ------------------------------------------------------------------
        df['Acc_Norm'] = np.linalg.norm(
            df[['Accelerometer x', 'Accelerometer y', 'Accelerometer z']].values,
            axis=1
        )
        df['Gyro_Norm'] = np.linalg.norm(
            df[['Gyroscope x', 'Gyroscope y', 'Gyroscope z']].values,
            axis=1
        )

        return df
