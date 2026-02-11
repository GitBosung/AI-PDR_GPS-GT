import numpy as np
import pandas as pd
import logging
from pyproj import Proj
from scipy.interpolate import PchipInterpolator
import math
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import butter, filtfilt


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
    def _lowpass_filter(df, cols, fs=50.0, cutoff=5.0, order=4):
        """
        Butterworth 저역통과필터 적용

        fs: 샘플링 주파수 (Hz)
        cutoff: 컷오프 주파수 (Hz)
        order: 필터 차수
        """
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq

        b, a = butter(order, normal_cutoff, btype="low", analog=False)

        for c in cols:
            x = df[c].astype(float).to_numpy()

            # 너무 짧으면 필터 생략
            if len(x) < order * 3:
                continue

            df[c] = filtfilt(b, a, x)

        return df


    def load_and_preprocess_csv(
        file_path, skiprows=100, skipfooter=100, flag=False, zone=52, window_size=200
    ):
        if flag:
            df = pd.read_csv(
                file_path,
                skiprows=skiprows,
                skipfooter=skipfooter,
                na_values=["", "nan", "NaN"],
                engine="python",
            ).fillna(0)
        else:
            df = pd.read_csv(
                file_path, skiprows=skiprows, skipfooter=skipfooter, engine="python"
            )

        df.columns = [
            "Time",
            "Accelerometer x",
            "Accelerometer y",
            "Accelerometer z",
            "Gyroscope x",
            "Gyroscope y",
            "Gyroscope z",
            "Magnetometer x",
            "Magnetometer y",
            "Magnetometer z",
            "Orientation x",
            "Orientation y",
            "Orientation z",
            "Pressure",
            "Latitude",
            "Longitude",
            "Altitude",
            "Speed_GPS",
        ]

        df["Time"] = pd.to_datetime(df["Time"], format="%Y-%m-%d %H:%M:%S.%f")
        start_dt = df["Time"].iloc[0]
        df["Elapsed Time"] = (df["Time"] - start_dt).dt.total_seconds()

        # df["Acc_Norm"] = np.linalg.norm(
        #     df[["Accelerometer x", "Accelerometer y", "Accelerometer z"]].values, axis=1
        # )

        acc_cols = ["Accelerometer x", "Accelerometer y", "Accelerometer z"]
        gyro_cols = ["Gyroscope x", "Gyroscope y", "Gyroscope z"]
        
        df = DataProcessor._lowpass_filter(
            df,
            cols=gyro_cols,
            fs=50.0,     
            cutoff=3.0,  
            order=2
        )
        
        # df = DataProcessor._lowpass_filter(
        #     df,
        #     cols=acc_cols,
        #     fs=50.0,     
        #     cutoff=3.0,  
        #     order=2
        # )

        df["Acc_Norm"] = np.linalg.norm(df[acc_cols].values, axis=1)
        df["Gyro_Norm"] = np.linalg.norm(df[gyro_cols].values, axis=1)

        e, n, df = DataProcessor.llh_to_enu(df, flag, zone)
        v_10hz, dh_10Hz = DataProcessor.interpol_vAndh(e, n)
        X, Y = DataProcessor.makeXY(df, v_10hz, dh_10Hz, window_size=window_size)

        return df, X, Y

    @staticmethod
    def llh_to_enu(df, flag, zone=52):
        if flag:
            valid_gps_mask = (
                df["Latitude"].notna()
                & df["Longitude"].notna()
                & (df["Latitude"].astype(str).str.strip() != "")
                & (df["Longitude"].astype(str).str.strip() != "")
            )
            valid_lat = pd.to_numeric(
                df.loc[valid_gps_mask, "Latitude"], errors="coerce"
            )
            valid_lon = pd.to_numeric(
                df.loc[valid_gps_mask, "Longitude"], errors="coerce"
            )
            final_mask = valid_lat.notna() & valid_lon.notna()
            valid_lat = valid_lat[final_mask].values
            valid_lon = valid_lon[final_mask].values
            if len(valid_lat) == 0:
                raise ValueError("유효한 GPS 데이터가 없습니다.")
            proj_enu = Proj(proj="utm", zone=zone, ellps="WGS84", south=False)
            e0, n0 = proj_enu(valid_lon[0], valid_lat[0])
            e_valid, n_valid = proj_enu(valid_lon, valid_lat)
            e_valid -= e0
            n_valid -= n0
            df["E"], df["N"] = np.nan, np.nan
            df.loc[valid_gps_mask, "E"] = e_valid
            df.loc[valid_gps_mask, "N"] = n_valid
            e = df["E"][df["E"].notna()].values
            n = df["N"][df["N"].notna()].values

        else:
            df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
            df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
            proj_enu = Proj(proj="utm", zone=zone, ellps="WGS84", south=False)
            e_all, n_all = proj_enu(df["Longitude"].values, df["Latitude"].values)
            e0, n0 = proj_enu(df["Longitude"].iloc[0], df["Latitude"].iloc[0])
            df["E"] = e_all - e0
            df["N"] = n_all - n0
            M = len(df)
            n_sec = M // 50
            e, n = [], []
            for i in range(n_sec):
                idx = min(i * 50 + 25, M - 1)
                e.append(df["E"].iloc[idx])
                n.append(df["N"].iloc[idx])
            e = np.array(e)
            n = np.array(n)

        # mask = mask_speed & mask_heading
        delta_e = np.diff(e)
        delta_n = np.diff(n)
        step = np.hypot(delta_e, delta_n)           # len = L-1

        # 3) 정지 구간
        stop_mask = step < 0.05

        # 4) 최종 마스크 (모두 길이 L-1)
        mask = (~stop_mask)
        gps_idx_all = np.arange(1, len(e))
        bad_idx = gps_idx_all[~mask]

        print(f"총 GPS 샘플: {len(e)}, 이상치 GPS 샘플: {len(bad_idx)}")

        # -------------------------------------------------------
        # (4) 센서데이터 블록 drop (1Hz GPS → 50Hz 센서)
        # -------------------------------------------------------
        drop_idx = []
        for gi in bad_idx:
            start = gi * 50
            end = (gi + 1) * 50
            drop_idx.extend(range(start, min(end, len(df))))
        df = df.drop(drop_idx).reset_index(drop=True)

        if flag:
            # --- NaN 있는 경우: 유효 GPS만 모아서 다시 ENU trajectory 계산 ---
            valid_gps_mask = (
                df["Latitude"].notna()
                & df["Longitude"].notna()
                & (df["Latitude"].astype(str).str.strip() != "")
                & (df["Longitude"].astype(str).str.strip() != "")
            )
            valid_lat = pd.to_numeric(
                df.loc[valid_gps_mask, "Latitude"], errors="coerce"
            )
            valid_lon = pd.to_numeric(
                df.loc[valid_gps_mask, "Longitude"], errors="coerce"
            )
            final_mask = valid_lat.notna() & valid_lon.notna()
            valid_lat = valid_lat[final_mask].values
            valid_lon = valid_lon[final_mask].values

            if len(valid_lat) < 2:
                raise ValueError("유효 GPS가 drop 이후 2개 미만으로 남음")

            proj_enu = Proj(proj="utm", zone=zone, ellps="WGS84", south=False)
            e0, n0 = proj_enu(valid_lon[0], valid_lat[0])
            e_valid, n_valid = proj_enu(valid_lon, valid_lat)
            e, n = e_valid - e0, n_valid - n0

        else:
            # --- NaN 없는 경우: 센서 50Hz 중간 샘플 뽑기 ---
            M = len(df)
            n_sec = M // 50
            e, n = [], []
            for i in range(n_sec):
                idx = min(i * 50 + 25, M - 1)
                e.append(df["E"].iloc[idx])
                n.append(df["N"].iloc[idx])
            e, n = np.array(e), np.array(n)

        # # -------------------------------------------------------
        # # (5) 초기 heading 정렬 + 보간
        # # -------------------------------------------------------
        dx0, dy0 = e[1] - e[0], n[1] - n[0]
        theta0 = math.atan2(dy0, dx0)
        R0 = np.array(
            [
                [math.cos(-theta0), -math.sin(-theta0)],
                [math.sin(-theta0), math.cos(-theta0)],
            ]
        )
        coords = np.vstack([e - e[0], n - n[0]])
        rotated = R0 @ coords
        e_corr, n_corr = rotated[0], rotated[1]

        return e_corr, n_corr, df
    
    @staticmethod
    def interpol_vAndh(e, n):
        delta_e = np.diff(e)
        delta_n = np.diff(n)

        v_1hz = (delta_e**2 + delta_n**2) ** 0.5
        heading_1hz = np.arctan2(delta_n, delta_e)
        dh_1hz = np.diff(np.unwrap(heading_1hz))
        origin_v = v_1hz[1:]
        origin_dh = dh_1hz

        N = len(origin_v)  # 1Hz 샘플 수
        t = np.arange(N, dtype=float)  # 0..N-1
        t_new = np.linspace(0.0, N - 1, (N - 1) * 10 + 1)  # ✅ 10Hz로 0~N-1초, 총 (N-1)*10+1개
        #t_new = np.linspace(0.0, N - 1, (N - 1) * 50 + 1)  # ✅ 50Hz로 0~N-1초, 총 (N-1)*50+1개
        pv = PchipInterpolator(t, origin_v)
        pdh = PchipInterpolator(t, origin_dh)
        
        v_10Hz = pv(t_new)
        dh_10Hz = pdh(t_new)

        return v_10Hz, dh_10Hz

    @staticmethod
    def makeXY(df, v_10Hz, dh_10Hz, window_size):
        stride = 5
        sensor_cols = [
            "Accelerometer x",
            "Accelerometer y",
            "Accelerometer z",
            "Gyroscope x",
            "Gyroscope y",
            "Gyroscope z",
            #"Acc_Norm",
            #"Gyro_Norm",
        ]

        values = df[sensor_cols].to_numpy()   # (N, num_features)
        N, num_features = values.shape

        # -------------------------
        # 1) X 만들기 (stride에 따라 분기)
        # -------------------------
        if stride == 1:
            # sliding_window_view로 오버헤드 최소화
            # 결과 shape: (N - window_size + 1, window_size, num_features)
            windows = sliding_window_view(values, (window_size, num_features))
            X = windows[:, 0, :, :]  # (N, window_size, F)
        else:
            # 기존 코드 그대로 (stride=5 등)
            X_list = []
            for i in range(0, N - window_size + 1, stride):
                window = values[i : i + window_size]   # (window_size, num_features)
                X_list.append(window)
            X = np.stack(X_list, axis=0)           # (num_windows, window_size, num_features)

        Y_v = []
        Y_dh = []

        offsets = [0, 10, 20, 30]  # 1초 간격 (10Hz 기준)
        #offsets = [0, 50, 100, 150]
        for i in range(len(dh_10Hz) - max(offsets)):
            # v: 1초 단위 4개를 합
            Y_v.append(
                v_10Hz[i + offsets[0]]
                + v_10Hz[i + offsets[1]]
                + v_10Hz[i + offsets[2]]
                + v_10Hz[i + offsets[3]]
            )

            Y_dh.append(
                dh_10Hz[i + offsets[0]]
                + dh_10Hz[i + offsets[1]]
                + dh_10Hz[i + offsets[2]]
                + dh_10Hz[i + offsets[3]]
            )  
          
        Y = np.stack([Y_v, Y_dh], axis=1)
        X = X[: len(Y)]
        return X, Y

    @staticmethod
    def load_and_preprocess_csv_test(file_path, skiprows=50):
        # ... 기존 테스트용 전처리 로직 그대로 유지 ...
        df = pd.read_csv(file_path, skiprows=skiprows, skipfooter=100, engine="python")

        df.columns = [
            "Time",
            "Accelerometer x",
            "Accelerometer y",
            "Accelerometer z",
            "Gyroscope x",
            "Gyroscope y",
            "Gyroscope z",
            "Magnetometer x",
            "Magnetometer y",
            "Magnetometer z",
            "Orientation x",
            "Orientation y",
            "Orientation z",
            "Pressure",
            "Latitude",
            "Longitude",
            "Altitude",
            "Speed_GPS",
        ]
        df["Time"] = pd.to_datetime(df["Time"], format="%Y-%m-%d %H:%M:%S.%f")
        start_dt = df["Time"].iloc[0]
        df["Elapsed Time"] = (df["Time"] - start_dt).dt.total_seconds()

        acc_cols = ["Accelerometer x", "Accelerometer y", "Accelerometer z"]
        gyro_cols = ["Gyroscope x", "Gyroscope y", "Gyroscope z"]
        
        df = DataProcessor._lowpass_filter(
            df,
            cols=gyro_cols,
            fs=50.0,
            cutoff=3.0,
            order=2,
        )
        
        # df = DataProcessor._lowpass_filter(
        #                 df,
        #     cols=acc_cols,
        #     fs=50.0,
        #     cutoff=1.0,
        #     order=2,
        # )


        df["Acc_Norm"] = np.linalg.norm(
            df[["Accelerometer x", "Accelerometer y", "Accelerometer z"]].values, axis=1
        )
        df["Gyro_Norm"] = np.linalg.norm(
            df[["Gyroscope x", "Gyroscope y", "Gyroscope z"]].values, axis=1
        )

        return df