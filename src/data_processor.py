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
    def __init__(self):
        pass

    @staticmethod
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

        df["Acc_Norm"] = np.linalg.norm(
            df[["Accelerometer x", "Accelerometer y", "Accelerometer z"]].values, axis=1
        )
        df["Gyro_Norm"] = np.linalg.norm(
            df[["Gyroscope x", "Gyroscope y", "Gyroscope z"]].values, axis=1
        )

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
            # return e, n
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
            # return e, n

        # -------------------------------------------------------
        # (3) Outlier 탐지
        # -------------------------------------------------------
        use_iqr = True
        dx = np.diff(e)
        dy = np.diff(n)
        speed = np.hypot(dx, dy)
        heading = np.arctan2(dy, dx)

        def remove_outliers_iqr(values, k=3.0):
            q1, q3 = np.percentile(values, [25, 75])
            iqr = q3 - q1
            return (values >= q1 - k * iqr) & (values <= q3 + k * iqr)

        def remove_outliers_zscore(values, threshold=3.0):
            mean, std = np.mean(values), np.std(values)
            return np.abs((values - mean) / (std + 1e-8)) < threshold

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
    
    
    def fix_spikes_mean_neighbors(x, z_thresh=3.0, mean_dev_ratio=2.5, win=5):
        """
        x: 1D np.array
        z_thresh: 중앙값(Median)±MAD 기반 임계값 배수 (3.0 권장)
        mean_dev_ratio: 앞뒤 평균 대비 |x[i]-m| > mean_dev_ratio * (|x[i-1]-m|+|x[i+1]-m|)/2 형태의 추가 조건
                        (or 단순히 |x[i]-m| > mean_dev_ratio * local_std 로 바꿔도 됨)
        win: 롤링 중앙값/ MAD 윈도우(홀수 권장)

        반환: 보정된 x, 스파이크 마스크(원소가 True면 보정됨)
        """
        x = np.asarray(x, dtype=float).copy()
        n = len(x)
        if n < 3:
            return x, np.zeros(n, dtype=bool)

        # 롤링 중앙값과 MAD(중앙절대편차)
        s = pd.Series(x)
        med = s.rolling(win, center=True, min_periods=1).median().to_numpy()
        mad = s.rolling(win, center=True, min_periods=1).apply(
            lambda v: np.median(np.abs(v - np.median(v))), raw=True
        ).to_numpy()
        mad = np.where(mad < 1e-12, 1e-12, mad)  # 0 회피
        # z-score 유사 척도 (1.4826은 MAD→표준편차 보정 상수)
        z_like = np.abs(x - med) / (1.4826 * mad)

        # 이웃 평균 기준
        # m[i] = (x[i-1] + x[i+1]) / 2
        m = np.zeros_like(x)
        m[1:-1] = (x[:-2] + x[2:]) / 2
        m[0] = x[1]             # 맨 앞: 바로 다음 값으로
        m[-1] = x[-2]           # 맨 뒤: 바로 이전 값으로

        # 이웃 평균 대비 편차
        dev = np.abs(x - m)
        # 지역 스케일(간단히 앞뒤의 절댓편차 평균)
        local_scale = np.zeros_like(x)
        local_scale[1:-1] = (np.abs(x[1:-1] - x[:-2]) + np.abs(x[1:-1] - x[2:])) / 2
        local_scale = np.where(local_scale < 1e-6, 1e-6, local_scale)

        # 두 조건 중 하나라도 만족하면 스파이크로 간주
        spike_mask = (z_like > z_thresh) | (dev / local_scale > mean_dev_ratio)

        # 앞뒤 평균으로 대체 (엣지는 위에서 정의한 m 사용)
        x_fixed = x.copy()
        x_fixed[spike_mask] = m[spike_mask]

        return x_fixed, spike_mask

    @staticmethod
    def interpol_vAndh(e, n):
        delta_e = np.diff(e)
        delta_n = np.diff(n)

        v_1hz = np.hypot(delta_e, delta_n)
        v_1hz = (delta_e**2 + delta_n**2) ** 0.5
        v_1hz, _ = DataProcessor.fix_spikes_mean_neighbors(v_1hz, z_thresh=3.0, mean_dev_ratio=2.5, win=5)
        dh_1hz = np.diff(np.unwrap(np.arctan2(delta_n, delta_e)))

        origin_v = v_1hz[1:]
        origin_dh = dh_1hz

        N = len(origin_v)  # 1Hz 샘플 수
        t = np.arange(N, dtype=float)  # 0..N-1
        t_new = np.linspace(
            0.0, N - 1, (N - 1) * 10 + 1
        )  # ✅ 10Hz로 0~N-1초, 총 (N-1)*10+1개
        # t_new = np.linspace(0.0, N-1, (N-1)*50 + 1)  # ✅ 10Hz로 0~N-1초, 총 (N-1)*10+1개
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
            "Acc_Norm",
            "Gyro_Norm",
        ]

        X = []
        for i in range(0, len(df) - window_size + 1, stride):
            window = (
                df[sensor_cols].iloc[i : i + window_size].values
            )  # (200, 8) numpy array
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

        # offsets = [0, 10, 20]  # 1초 간격 (10Hz 기준)
        # for i in range(len(dh_10Hz) - max(offsets)):
        #     # v: 1초 단위 4개를 합
        #     Y_v.append(
        #         v_10Hz[i + offsets[0]] + v_10Hz[i + offsets[1]] + v_10Hz[i + offsets[2]]
        #     )

        #     Y_dh.append(
        #         dh_10Hz[i + offsets[0]]
        #         + dh_10Hz[i + offsets[1]]
        #         + dh_10Hz[i + offsets[2]]
        #     )
        # Y = np.stack([Y_v, Y_dh], axis=1)
        # X = X[: len(Y)]
        return X, Y

    @staticmethod
    def load_and_preprocess_csv_test(file_path, skiprows=50):
        # ... 기존 테스트용 전처리 로직 그대로 유지 ...
        df = pd.read_csv(file_path, skiprows=skiprows, skipfooter=10, engine="python")

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

        df["Acc_Norm"] = np.linalg.norm(
            df[["Accelerometer x", "Accelerometer y", "Accelerometer z"]].values, axis=1
        )
        df["Gyro_Norm"] = np.linalg.norm(
            df[["Gyroscope x", "Gyroscope y", "Gyroscope z"]].values, axis=1
        )

        return df
