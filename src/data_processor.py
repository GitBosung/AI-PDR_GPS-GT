import numpy as np
import pandas as pd
import os
import logging
from datetime import datetime
from pyproj import Proj
from scipy.interpolate import interp1d
from scipy.interpolate import PchipInterpolator
import math
import matplotlib.pyplot as plt 


logger = logging.getLogger(__name__)

# -----------------------------------------------------------
# 전역(글로벌) Proj 객체: UTM zone 52N (EPSG:32652)
# 위경도 좌표를 UTM(동-북) 좌표로 변환하기 위해 사용
# -----------------------------------------------------------
_proj_utm52 = Proj("epsg:32652")


class DataProcessor:
    def __init__(self, window_size=200):
        self.window_size = window_size
        
    @staticmethod
    def load_and_preprocess_csv(file_path, skiprows=100, skipfooter=100, flag=False, zone=52, window_size=200):
        if flag:
            df = pd.read_csv(file_path, skiprows=skiprows, skipfooter=skipfooter,
                             na_values=['', 'nan', 'NaN'], engine='python').fillna(0)
        else:
            df = pd.read_csv(file_path, skiprows=skiprows, skipfooter=skipfooter, engine='python')
            
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
        
        e, n = DataProcessor.llh_to_enu(df, flag, zone)
        v_10hz, dh_10Hz = DataProcessor.interpol_vAndh(e, n)
        X, Y = DataProcessor.makeXY(df, v_10hz, dh_10Hz, window_size=window_size)
        
        return df, X, Y
    
    @staticmethod
    def llh_to_enu(df, flag, zone=52):
        if flag:
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
            return e, n
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
            return e, n
            
    @staticmethod
    def interpol_vAndh(e, n):
        delta_e = np.diff(e)
        delta_n = np.diff(n)
        
        v_1hz = np.hypot(delta_e, delta_n)

        theta = np.arctan2(delta_n, delta_e)                     # 1Hz 혹은 원 샘플 기준 각
        theta_u = np.unwrap(theta, discont=np.pi/2)              # ← 먼저 언랩(더 깐깐: π/2)
        dh = np.diff(theta_u)                                    # ← 그 다음 차분

        # 저속 구간 마스킹(권장)
        eps = 1e-2  # 단위에 맞게 조정(예: m/step)
        mask = (v_1hz[:-1] > eps) & (v_1hz[1:] > eps)
        dh_1hz = np.where(mask, dh, 0.0)                       # 0 또는 보간 처리
        
        # v_1hz = (delta_e**2 + delta_n**2)**0.5
        # dh_1hz = np.diff(np.unwrap(np.arctan2(delta_n, delta_e)))
        
        origin_v = v_1hz[1:]
        origin_dh = dh_1hz

        N = len(origin_v)                  # 1Hz 샘플 수
        t = np.arange(N, dtype=float)      # 0..N-1
        t_new = np.linspace(0.0, N-1, (N-1)*10 + 1)  # ✅ 10Hz로 0~N-1초, 총 (N-1)*10+1개
        # t_new = np.linspace(0.0, N-1, (N-1)*50 + 1)  # ✅ 10Hz로 0~N-1초, 총 (N-1)*10+1개

        pv  = PchipInterpolator(t, origin_v)
        pdh = PchipInterpolator(t, origin_dh)
        v_10Hz  = pv(t_new)
        dh_10Hz = pdh(t_new)
        
        return v_10Hz, dh_10Hz
    
    @staticmethod
    def makeXY(df, v_10Hz, dh_10Hz, window_size):
        stride = 5
        sensor_cols = ['Accelerometer x','Accelerometer y','Accelerometer z',
                'Gyroscope x','Gyroscope y','Gyroscope z',
                'Acc_Norm', 'Gyro_Norm']

        X = []
        for i in range(0, len(df) - window_size + 1, stride):
            window = df[sensor_cols].iloc[i:i+window_size].values  # (200, 8) numpy array
            X.append(window)

        X = np.array(X)  # (n, 200, 8)

        Y_v = []
        Y_dh = []

        offsets = [0, 10, 20, 30]   # 1초 간격 (10Hz 기준)
        for i in range(len(dh_10Hz) - max(offsets)):
            # v: 1초 단위 4개를 합
            Y_v.append(v_10Hz[i + offsets[0]] +
                    v_10Hz[i + offsets[1]] +
                    v_10Hz[i + offsets[2]] +
                    v_10Hz[i + offsets[3]])

            Y_dh.append(dh_10Hz[i + offsets[0]] +
                    dh_10Hz[i + offsets[1]] +
                    dh_10Hz[i + offsets[2]] +
                    dh_10Hz[i + offsets[3]])  # 10Hz × 40 = 4초
        Y = np.stack([Y_v, Y_dh], axis=1)
        X = X[:len(Y)]
        return X, Y
    
    
    @staticmethod
    def load_and_preprocess_csv_test(file_path, skiprows=50):
        # ... 기존 테스트용 전처리 로직 그대로 유지 ...
        df = pd.read_csv(
        file_path, skiprows=skiprows, skipfooter=10, engine='python'
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



