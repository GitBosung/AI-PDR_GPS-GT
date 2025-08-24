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
    def load_and_preprocess_csv(file_path, skiprows=500, flag=False, zone=52):
        
        window_size = 200
        if flag:
            df = pd.read_csv(
                file_path, skiprows=skiprows, skipfooter=200,
                na_values=['', 'nan', 'NaN'], engine='python'
            ).fillna(0)
        else:
            df = pd.read_csv(
            file_path, skiprows=skiprows, skipfooter=200, engine='python'
            )
            
        df.columns = [
            'Time',
            'Accelerometer x', 'Accelerometer y', 'Accelerometer z',
            'Gyroscope x', 'Gyroscope y', 'Gyroscope z',
            'Magnetometer x', 'Magnetometer y', 'Magnetometer z',
            'Orientation x', 'Orientation y', 'Orientation z',
            'Pressure', 'Latitude', 'Longitude', 'Altitude', 'Speed_GPS'
        ]
        
        df['Acc_Norm'] = np.sqrt(df['Accelerometer x']**2 +
                                 df['Accelerometer y']**2 +
                                 df['Accelerometer z']**2)
        
        df['Gyro_Norm'] = np.sqrt(df['Gyroscope x']**2 +
                                  df['Gyroscope y']**2 +
                                  df['Gyroscope z']**2)
        

        df['Time'] = pd.to_datetime(df['Time'], format='%Y-%m-%d %H:%M:%S.%f')
        start_dt = df['Time'].iloc[0]
        df['Elapsed Time'] = (df['Time'] - start_dt).dt.total_seconds()
        
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
                
        # ------------------------------------------------------------------
        # 5) 1Hz 궤적 보정: 초기 heading 회전 정렬
        # ------------------------------------------------------------------
        dx0, dy0 = e[1]-e[0], n[1]-n[0]
        theta0 = math.atan2(dy0, dx0)
        R0 = np.array([[math.cos(-theta0), -math.sin(-theta0)],
                    [math.sin(-theta0),  math.cos(-theta0)]])
        coords = np.vstack([e-e[0], n-n[0]])
        rotated = R0 @ coords
        e_corr, n_corr = rotated[0], rotated[1]
        
        # 1) 보간용 원본 인덱스와 타겟 인덱스 생성
        M = len(e_corr)
        t_old = np.arange(M)                          
        t_new = np.linspace(0, M - 1, (M-1)*50 + 1)          

        # # 2) 선형 보간 함수 생성
        # f_e = interp1d(t_old, e_corr, kind='cubic')
        # f_n = interp1d(t_old, n_corr, kind='cubic')

        # # 3) 50개 포인트로 보간
        # e_aug = f_e(t_new)   # shape = (50,)
        # n_aug = f_n(t_new)   # shape = (50,)
        
        f_e = PchipInterpolator(t_old.astype(float), e_corr.astype(float), extrapolate=True)
        f_n = PchipInterpolator(t_old.astype(float), n_corr.astype(float), extrapolate=True)

        e_aug = f_e(t_new.astype(float))
        n_aug = f_n(t_new.astype(float))
        
        # df['e_aug'] = e_aug
        # df['n_aug'] = n_aug
        
        df2 = df.iloc[:len(e_aug)].copy()

        # 2) 그 위에 보간 결과를 할당
        df2['e_aug'] = e_aug
        df2['n_aug'] = n_aug
        
        e = df2['e_aug'].values  # shape = (T,)
        n = df2['n_aug'].values  # shape = (T,)

         
        num_wins = len(e) - window_size + 1

        # 결과 저장용
        y_speed = np.zeros(num_wins)
        y_dh    = np.zeros(num_wins)
        
        # 5) Norm 채널 추가
        df2['Acc_Norm']  = np.linalg.norm(
            df2[['Accelerometer x','Accelerometer y','Accelerometer z']].values,
            axis=1
        )
        df2['Gyro_Norm'] = np.linalg.norm(
            df2[['Gyroscope x','Gyroscope y','Gyroscope z']].values,
            axis=1
        )
        

        # for i in range(num_wins):
        #     we = e[i : i+window_size]   # window of Eastings
        #     wn = n[i : i+window_size]   # window of Northings

        #     # 1) 속도: 시작점→종료점 직선 이동 거리 (m/s)
        #     dx = we[-1] - we[0]
        #     dy = wn[-1] - wn[0]
        #     dist = np.hypot(dx, dy)    # sqrt(dx^2 + dy^2)
        #     y_speed[i] = dist          # 1초 동안 이동거리 = 속도(m/s)

        #     # 2) 방향변화량: 각 샘플 간 heading 변화를 누적
            
        #     headings = np.arctan2(np.diff(wn), np.diff(we))
        #     headings = np.unwrap(headings)

        #     # Δheading 원본
        #     raw_delta = headings[-1] - headings[0]
            
        #     dh = np.diff(headings)
        #     total_delta = np.sum(dh)

        #     # # 소프트 데드존 적용
        #     # if abs(raw_delta) < threshold:
        #     #     delta = 0
        #     # else:
        #     #     delta = np.sign(raw_delta) * (abs(raw_delta) - threshold)

        #     y_dh[i] = total_delta
            
        for i in range(num_wins):
            we = e[i : i+window_size]   # window of Eastings
            wn = n[i : i+window_size]   # window of Northings

            # 1) 속도: 시작점→종료점 직선 이동 거리 (m)
            dx = we[-1] - we[0]
            dy = wn[-1] - wn[0]
            dist = np.hypot(dx, dy)
            y_speed[i] = dist

            # 2) 방향변화량: 벡터 간 회전각 누적 (언랩 불필요)
            vel_x = np.diff(we)   # ΔE
            vel_y = np.diff(wn)   # ΔN
            spd   = np.hypot(vel_x, vel_y)

            EPS = 0.02  # 데이터 스케일에 맞게 조정(예: 1~5cm 이동 임계)
            valid = spd > EPS

            if np.count_nonzero(valid) >= 3:
                vx = vel_x[valid]
                vy = vel_y[valid]

                # (k -> k+1) 회전각: atan2(cross, dot) ∈ (-π, π)
                cross = vx[:-1]*vy[1:] - vy[:-1]*vx[1:]
                dot   = vx[:-1]*vx[1:] + vy[:-1]*vy[1:]
                delta_angles = np.arctan2(cross, dot)

                total_delta = np.sum(delta_angles)

                # 최종 정규화 선택 (원하면 사용)
                total_delta = (total_delta + np.pi) % (2*np.pi) - np.pi
            else:
                total_delta = 0.0

            y_dh[i] = total_delta
            
        sensor_cols = ['Accelerometer x','Accelerometer y','Accelerometer z',
                        'Gyroscope x','Gyroscope y','Gyroscope z',
                        'Acc_Norm',
                        'Gyro_Norm']
        
        sensor = df2[sensor_cols].values  # shape = (T,6)

        X = np.zeros((num_wins, window_size, len(sensor_cols)), dtype=np.float32)
        for i in range(num_wins):
            X[i] = sensor[i : i+window_size]

        # (2) Y 결합
        Y = np.vstack([y_speed, y_dh]).T   # shape = (num_wins, 2)
                
        return df2, X, Y
    
    
        # M = len(df)
        # sensor_windows = []
        # for i in range(0, M-50+1):
        #     arr = np.stack([
        #         df['Accelerometer x'].values[i:i+50],
        #         df['Accelerometer y'].values[i:i+50],
        #         df['Accelerometer z'].values[i:i+50],
        #         df['Gyroscope x'].values[i:i+50],
        #         df['Gyroscope y'].values[i:i+50],
        #         df['Gyroscope z'].values[i:i+50],
        #         df['Acc_Norm'].values[i:i+50],
        #         df['Gyro_Norm'].values[i:i+50],
        #     ], axis=1)  # (50,8)


        #     sensor_windows.append(arr)

        # # ------------------------------------------------------------------
        # # 9–10) X, Y 생성
        # # ------------------------------------------------------------------
        # usable = min(len(sensor_windows), len(speed_1), len(heading_1))
        # X = np.stack(sensor_windows[:usable], axis=0)  # (usable,50,38)
        # Y = np.stack([speed_1[:usable], heading_1[:usable]], axis=1)

        # # ------------------------------------------------------------------
        # # 11) 원본 df에 speed_1, heading_1 컬럼 추가
        # # ------------------------------------------------------------------
        # df['speed_1']   = np.nan
        # df['heading_1'] = np.nan
        # for j in range(usable):
        #     df.at[j, 'speed_1']   = speed_1[j]
        #     df.at[j, 'heading_1'] = heading_1[j]

        # return df, X, Y

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
