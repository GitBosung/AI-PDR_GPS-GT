import numpy as np
import pandas as pd
import os
import logging
from datetime import datetime
from pyproj import Proj
from scipy.interpolate import interp1d
from scipy.interpolate import PchipInterpolator
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
      - 스무딩(이동평균) 적용
      - 윈도우별 통계(평균·표준편차·분산·최대·최소) 추가
      - 위경도 → ENU 좌표 변환
      - 1Hz 기준 궤적 계산 및 보간
      - 50Hz 센서 윈도우 생성 (원본 8채널 + 통계 30채널 = 총 38채널)
      - 슬라이딩 윈도우 X, Y 데이터 반환
    """

    def __init__(self):
        pass

    @staticmethod
    def load_and_preprocess_csv(file_path, skiprows=500, flag=False, zone=52, window_size=200):
        # window_size = 200
        if flag:
            df = pd.read_csv(
                file_path, skiprows=skiprows, skipfooter=500,
                na_values=['', 'nan', 'NaN'], engine='python'
            ).fillna(0)
        else:
            df = pd.read_csv(
            file_path, skiprows=skiprows, skipfooter=500, engine='python'
            )
            
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
        
        df['Acc_Norm'] = np.linalg.norm(df[['Accelerometer x', 'Accelerometer y', 'Accelerometer z']].values, axis=1)
        df['Gyro_Norm'] = np.linalg.norm(df[['Gyroscope x', 'Gyroscope y', 'Gyroscope z']].values, axis=1)

        if flag:
            # (기존 원본 로직 그대로 유지)
            valid_gps_mask = (
                df['Latitude'].notna() & df['Longitude'].notna() &
                (df['Latitude'].astype(str).str.strip() != '') &
                (df['Longitude'].astype(str).str.strip() != '')
            )
            valid_lat = pd.to_numeric(df.loc[valid_gps_mask, 'Latitude'], errors='coerce')
            valid_lon = pd.to_numeric(df.loc[valid_gps_mask, 'Longitude'], errors='coerce')
            final_mask = valid_lat.notna() & valid_lon.notna()
            valid_lat = valid_lat[final_mask].values
            valid_lon = valid_lon[final_mask].values
            if len(valid_lat) == 0:
                raise ValueError("유효한 GPS 데이터가 없습니다.")
            proj_enu = Proj(proj='utm', zone=zone, ellps='WGS84', south=False)
            e0, n0 = proj_enu(valid_lon[0], valid_lat[0])
            e_valid, n_valid = proj_enu(valid_lon, valid_lat)
            e_valid -= e0; n_valid -= n0
            df['E'], df['N'] = np.nan, np.nan
            df.loc[valid_gps_mask, 'E'] = e_valid
            df.loc[valid_gps_mask, 'N'] = n_valid
            e = df['E'][df['E'].notna()].values
            n = df['N'][df['N'].notna()].values
        else:
            df['Latitude']  = pd.to_numeric(df['Latitude'],  errors='coerce')
            df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
            proj_enu = Proj(proj='utm', zone=zone, ellps='WGS84', south=False)
            e_all, n_all = proj_enu(df['Longitude'].values, df['Latitude'].values)
            e0, n0 = proj_enu(df['Longitude'].iloc[0], df['Latitude'].iloc[0])
            df['E'] = e_all - e0
            df['N'] = n_all - n0
            M = len(df)
            n_sec = M // 50
            e, n = [], []
            for i in range(n_sec):
                idx = min(i*50 + 25, M-1)
                e.append(df['E'].iloc[idx])
                n.append(df['N'].iloc[idx])
            e = np.array(e)
            n = np.array(n)
                
        # -------------------------------------------------------
        # (3) Outlier 탐지
        # -------------------------------------------------------
        use_iqr=True
        dx = np.diff(e); dy = np.diff(n)
        speed = np.hypot(dx, dy)
        heading = np.arctan2(dy, dx)

        def remove_outliers_iqr(values, k=1.5):
            q1, q3 = np.percentile(values, [25, 75])
            iqr = q3 - q1
            return (values >= q1 - k*iqr) & (values <= q3 + k*iqr)

        def remove_outliers_zscore(values, threshold=3.0):
            mean, std = np.mean(values), np.std(values)
            return np.abs((values-mean)/(std+1e-8)) < threshold

        if use_iqr:
            mask_speed = remove_outliers_iqr(speed)
            mask_heading = remove_outliers_iqr(np.degrees(heading))
        else:
            mask_speed = remove_outliers_zscore(speed)
            mask_heading = remove_outliers_zscore(np.degrees(heading))

        mask = mask_speed & mask_heading
        gps_idx_all = np.arange(1, len(e))
        bad_idx = gps_idx_all[~mask]

        # -------------------------------------------------------
        # (4) 센서데이터 블록 drop (1Hz GPS → 50Hz 센서)
        # -------------------------------------------------------
        drop_idx = []
        for gi in bad_idx:
            start = gi*50
            end   = (gi+1)*50
            drop_idx.extend(range(start, min(end, len(df))))
        df = df.drop(drop_idx).reset_index(drop=True)
        
        if flag:
            # --- NaN 있는 경우: 유효 GPS만 모아서 다시 ENU trajectory 계산 ---
            valid_gps_mask = (
                df['Latitude'].notna() & df['Longitude'].notna() &
                (df['Latitude'].astype(str).str.strip() != '') &
                (df['Longitude'].astype(str).str.strip() != '')
            )
            valid_lat = pd.to_numeric(df.loc[valid_gps_mask, 'Latitude'], errors='coerce')
            valid_lon = pd.to_numeric(df.loc[valid_gps_mask, 'Longitude'], errors='coerce')
            final_mask = valid_lat.notna() & valid_lon.notna()
            valid_lat = valid_lat[final_mask].values
            valid_lon = valid_lon[final_mask].values
            
            if len(valid_lat) < 2:
                raise ValueError("유효 GPS가 drop 이후 2개 미만으로 남음")

            proj_enu = Proj(proj='utm', zone=zone, ellps='WGS84', south=False)
            e0, n0 = proj_enu(valid_lon[0], valid_lat[0])
            e_valid, n_valid = proj_enu(valid_lon, valid_lat)
            e, n = e_valid - e0, n_valid - n0

        else:
            # --- NaN 없는 경우: 센서 50Hz 중간 샘플 뽑기 ---
            M = len(df)
            n_sec = M // 50
            e, n = [], []
            for i in range(n_sec):
                idx = min(i*50 + 25, M-1)
                e.append(df['E'].iloc[idx])
                n.append(df['N'].iloc[idx])
            e, n = np.array(e), np.array(n)
        
        # # -------------------------------------------------------
        # # (5) 초기 heading 정렬 + 보간
        # # -------------------------------------------------------
        # dx0, dy0 = e[1]-e[0], n[1]-n[0]
        # theta0 = math.atan2(dy0, dx0)
        # R0 = np.array([[math.cos(-theta0), -math.sin(-theta0)],
        #             [math.sin(-theta0),  math.cos(-theta0)]])
        # coords = np.vstack([e-e[0], n-n[0]])
        # rotated = R0 @ coords
        # e_corr, n_corr = rotated[0], rotated[1]

        # delta_e = np.diff(e_corr)
        # delta_n = np.diff(n_corr)
        
        # speed_1hz = np.hypot(delta_e, delta_n)
        # heading_change_1hz = np.diff(np.unwrap(np.arctan2(delta_n, delta_e)))

        # N = len(speed_1hz)
        # N2 = len(heading_change_1hz)
        
        # t_old  = np.arange(N)
        # t_new  = np.linspace(0, N - 1, (N - 1) * 50 + 1)

        # t_old2 = np.arange(N2)   # hz1_v 길이에 맞춤
        # t_new2 = np.linspace(0, N2 - 1, (N2 - 1) * 50 + 1)

        # f_h_cubic = interp1d(t_old2, heading_change_1hz, kind='cubic', fill_value="extrapolate")
        # f_v_cubic = interp1d(t_old,  speed_1hz, kind='cubic', fill_value="extrapolate")

        # interp_h = f_h_cubic(t_new2)
        # interp_v = f_v_cubic(t_new)
        
        
        # add_num = window_size // 50
        
        # n_sec_speed = []
        # n_sec_dh = []
        
        # for i in range(len(interp_h)):
        #     n_sec_speed.append(interp_v[i:i+add_num].sum())
        #     n_sec_dh.append(interp_h[i:i+add_num].sum()/add_num)
            
        # n_sec_speed = np.array(n_sec_speed)
        # n_sec_dh = np.array(n_sec_dh)
             
        # num_wins = len(n_sec_speed) - window_size + 1
        
        # sensor_cols = ['Accelerometer x','Accelerometer y','Accelerometer z',
        #                 'Gyroscope x','Gyroscope y','Gyroscope z',
        #                 'Acc_Norm',
        #                 'Gyro_Norm']
        
        # sensor = df[sensor_cols].values  # shape = (T,6)

        # # X = np.zeros((num_wins, window_size, len(sensor_cols)), dtype=np.float32)
        # # for i in range(num_wins):
        # #     X[i] = sensor[i : i+window_size]

        # # Y = np.vstack([n_sec_speed, n_sec_dh]).T
        # num_wins_sensor = len(sensor) - window_size + 1
        # num_wins_label  = len(n_sec_speed) - window_size + 1
        # num_wins = min(num_wins_sensor, num_wins_label)

        # X = np.zeros((num_wins, window_size, len(sensor_cols)), dtype=np.float32)
        # for i in range(num_wins):
        #     X[i] = sensor[i : i+window_size]

        # Y_all = np.vstack([n_sec_speed, n_sec_dh]).T
        # Y = Y_all[window_size-1 : window_size-1+num_wins]
        
            # -------------------------------------------------------
        # (5) 초기 heading 정렬 + 보간
        # -------------------------------------------------------
        dx0, dy0 = e[1]-e[0], n[1]-n[0]
        theta0 = math.atan2(dy0, dx0)
        R0 = np.array([[math.cos(-theta0), -math.sin(-theta0)],
                    [math.sin(-theta0),  math.cos(-theta0)]])
        coords = np.vstack([e-e[0], n-n[0]])
        rotated = R0 @ coords
        e_corr, n_corr = rotated[0], rotated[1]

        delta_e = np.diff(e_corr)
        delta_n = np.diff(n_corr)
        
        speed_1hz = np.hypot(delta_e, delta_n)
        heading_change_1hz = np.diff(np.unwrap(np.arctan2(delta_n, delta_e)))
        n_sec_hc = []
        n_sec_speed = []
        add_num = window_size // 50
        
        for i in range(len(heading_change_1hz)):
            n_sec_speed.append(speed_1hz[i:i+add_num].sum())
            n_sec_hc.append(heading_change_1hz[i:i+add_num].sum())
        
        n_sec_speed = np.array(n_sec_speed)
        n_sec_hc = np.array(n_sec_hc)    
            
        N = len(n_sec_speed)
        N2 = len(n_sec_hc)
        
        t_old  = np.arange(N)
        t_new  = np.linspace(0, N - 1, (N - 1) * 50 + 1)

        t_old2 = np.arange(N2)   # hz1_v 길이에 맞춤
        t_new2 = np.linspace(0, N2 - 1, (N2 - 1) * 50 + 1)

        f_h_cubic = interp1d(t_old2, n_sec_hc, kind='cubic', fill_value="extrapolate")
        f_v_cubic = interp1d(t_old,  n_sec_speed, kind='cubic', fill_value="extrapolate")

        interp_h = f_h_cubic(t_new2)
        interp_v = f_v_cubic(t_new)
        
        
        # n_sec_speed = []
        # n_sec_dh = []
        
        # for i in range(len(interp_h)):
        #     n_sec_speed.append(interp_v[i:i+add_num].sum())
        #     n_sec_dh.append(interp_h[i:i+add_num].sum()/add_num)
            
        # n_sec_speed = np.array(n_sec_speed)
        # n_sec_dh = np.array(n_sec_dh)
    
        
                
        num_wins = len(n_sec_speed) - window_size + 1
        
        sensor_cols = ['Accelerometer x','Accelerometer y','Accelerometer z',
                        'Gyroscope x','Gyroscope y','Gyroscope z',
                        'Acc_Norm',
                        'Gyro_Norm']
        
        sensor = df[sensor_cols].values  # shape = (T,6)

        # X = np.zeros((num_wins, window_size, len(sensor_cols)), dtype=np.float32)
        # for i in range(num_wins):
        #     X[i] = sensor[i : i+window_size]

        # Y = np.vstack([n_sec_speed, n_sec_dh]).T
        num_wins_sensor = len(sensor) - window_size + 1
        num_wins_label  = len(interp_h) - window_size + 1
        num_wins = min(num_wins_sensor, num_wins_label)

        X = np.zeros((num_wins, window_size, len(sensor_cols)), dtype=np.float32)
        for i in range(num_wins):
            X[i] = sensor[i : i+window_size]

        Y = np.vstack([interp_v, interp_h]).T
        Y = Y[:num_wins]
                
        return df, X, Y
        
    #delimiter=',', header=0, 
    @staticmethod
    def load_and_preprocess_csv_test(file_path, skiprows=100):
        # ... 기존 테스트용 전처리 로직 그대로 유지 ...
        df = pd.read_csv(
        file_path, skiprows=skiprows, skipfooter=0, engine='python'
        )

        df.columns = [
            'Time',
            'Accelerometer x','Accelerometer y','Accelerometer z',
            'Gyroscope x','Gyroscope y','Gyroscope z',
            'Magnetometer x','Magnetometer y','Magnetometer z',
            'Orientation x','Orientation y','Orientation z',
            'Pressure','Latitude','Longitude','Altitude','Speed_GPS'
        ]
        df['Time'] = pd.to_datetime(df['Time'], format='%Y-%m-%d %H:%M:%S.%f')
        start_dt = df['Time'].iloc[0]
        df['Elapsed Time'] = (df['Time'] - start_dt).dt.total_seconds()

        df['Acc_Norm'] = np.linalg.norm(
            df[['Accelerometer x','Accelerometer y','Accelerometer z']].values, axis=1
        )
        df['Gyro_Norm'] = np.linalg.norm(
            df[['Gyroscope x','Gyroscope y','Gyroscope z']].values, axis=1
        )

        return df
