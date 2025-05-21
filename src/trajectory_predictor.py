import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class TrajectoryPredictor:
    """
    - 학습된 모델(model), sensor_scalers(dict of sklearn Scaler), window_size를 받아
      센서 데이터를 윈도우별로 스케일링 → 예측 → 시각화 수행
    """
    def __init__(self, model, sensor_scalers: dict, window_size=50):
        self.model = model
        self.sensor_scalers = sensor_scalers
        self.window_size = window_size

    def _prepare_windows(self, df: pd.DataFrame) -> np.ndarray:
        """
        DataFrame df에서 sensor_cols 열을 window_size 단위로 잘라
        각 윈도우마다 전역 scaler만 적용 후 3D numpy array로 반환
        (shape = [num_windows, window_size, num_features])
        """
        sensor_cols = [
            'Accelerometer x','Accelerometer y','Accelerometer z',
            'Gyroscope x','Gyroscope y','Gyroscope z',
            'Acc_Norm','Gyro_Norm',
        ]
        windows = []

        # non-overlap stride = window_size
        for start in range(0, len(df) - self.window_size + 1, 5):
            sub_df = df[sensor_cols].iloc[start : start + self.window_size].astype(np.float32)
            arr = sub_df.values  # (window_size, num_features)
            scaled = np.zeros_like(arr, dtype=np.float32)

            for idx, col in enumerate(sensor_cols):
                # 전역으로 학습된 scaler 적용
                scaler = self.sensor_scalers.get(col)
                if scaler:
                    scaled[:, idx] = scaler.transform(arr[:, idx].reshape(-1, 1)).ravel()
                else:
                    # scaler 없으면 원본 값을 그대로 사용
                    scaled[:, idx] = arr[:, idx]

            windows.append(scaled)
            
        return np.array(windows, dtype=np.float32)
            
    def _prepare_windows_1Hz(self, df: pd.DataFrame) -> np.ndarray:
        """
        DataFrame df에서 sensor_cols 열을 window_size 단위로 잘라
        각 윈도우마다 전역 scaler만 적용 후 3D numpy array로 반환
        (shape = [num_windows, window_size, num_features])
        """
        sensor_cols = [
            'Accelerometer x','Accelerometer y','Accelerometer z',
            'Gyroscope x','Gyroscope y','Gyroscope z',
            'Acc_Norm','Gyro_Norm',
        ]
        windows = []

        # non-overlap stride = window_size
        for start in range(0, len(df) - self.window_size + 1, 50):
            sub_df = df[sensor_cols].iloc[start : start + self.window_size].astype(np.float32)
            arr = sub_df.values  # (window_size, num_features)
            scaled = np.zeros_like(arr, dtype=np.float32)

            for idx, col in enumerate(sensor_cols):
                # 전역으로 학습된 scaler 적용
                scaler = self.sensor_scalers.get(col)
                if scaler:
                    scaled[:, idx] = scaler.transform(arr[:, idx].reshape(-1, 1)).ravel()
                else:
                    # scaler 없으면 원본 값을 그대로 사용
                    scaled[:, idx] = arr[:, idx]

            windows.append(scaled)

        return np.array(windows, dtype=np.float32)

    def predict_and_plot_trajectory(self, df: pd.DataFrame, plag_1Hz: bool = False):
        
        if plag_1Hz:
            X = self._prepare_windows_1Hz(df)               # (num_windows, window_size, num_features)
        else:
            X = self._prepare_windows(df)    
            # (num_windows, window_size, num_features)
        Y = self.model.predict(X) 
        
        if plag_1Hz:
            speeds = Y[:, 0] 
            dh     = Y[:, 1] 
        else:
            speeds = Y[:, 0] * 0.1
            dh     = Y[:, 1] * 0.1
            
        #X = self._prepare_windows(df)               # (num_windows, window_size, num_features)
            

        # Speed plot
        plt.figure(figsize=(8,4))
        plt.plot(speeds, '-o', label='Speed (m/s)', markersize=4)
        plt.title('Predicted Speed')
        plt.xlabel('Window Index'); plt.ylabel('Speed (m/s)')
        plt.grid(True); plt.legend(); plt.show()

        # Heading Change plot
        plt.figure(figsize=(8,4))
        plt.plot(np.degrees(dh), '-o', label='Heading Change (deg)', markersize=4)
        plt.title('Predicted Heading Change')
        plt.xlabel('Window Index'); plt.ylabel('Heading Change (deg)')
        plt.grid(True); plt.legend(); plt.show()

        # 누적 궤적 계산
        x = y = heading = 0.0
        traj_x, traj_y = [x], [y]
        for s, delta_h in zip(speeds, dh):
            heading += delta_h
            x += s * np.cos(heading)
            y += s * np.sin(heading)
            traj_x.append(x); traj_y.append(y)

        # 궤적 시각화
        plt.figure(figsize=(8,6))
        plt.plot(traj_x, traj_y, 'b-o', alpha=0.7, markersize=4, label='Predicted Path')
        plt.plot(traj_x[0], traj_y[0], 'go', markersize=8, label='Start')
        plt.plot(traj_x[-1], traj_y[-1], 'ro', markersize=8, label='End')
        plt.title('Predicted Movement Trajectory')
        plt.xlabel('Easting (m)'); plt.ylabel('Northing (m)')
        plt.grid(True); plt.axis('equal'); plt.legend(); plt.axis('equal'); plt.show()

        return Y, (traj_x, traj_y)

    def compare_trajectories(self, df: pd.DataFrame, plag_1Hz: bool = False):
        # 1) predict
        if plag_1Hz:
            X = self._prepare_windows_1Hz(df)
            stride = 50
        else:
            X = self._prepare_windows(df)
            stride = 5

        Y = self.model.predict(X)

        if plag_1Hz:
            pred_speed = Y[:, 0]
            pred_hc    = Y[:, 1]
        else:
            pred_speed = Y[:, 0] * 0.1
            pred_hc    = Y[:, 1] * 0.1

        # 2) GT 속도 변화를 stride 간격으로 샘플링해서 pred 길이에 맞추기
        gt_speed_full = df['dist_change'].values * 0.1
        # 윈도우 끝 지점 기준: stride-1, 2*stride-1, ...
        gt_speed_ws = gt_speed_full[stride - 1 :: stride]
        gt_speed_ws = gt_speed_ws[:len(pred_speed)]

        # 3) GT 절대 궤적 (E,N) -> 상대 좌표 (시작점 원점)
        E = df['E'].values; N = df['N'].values
        E_rel = E - E[0]; N_rel = N - N[0]
        
        E = df['E'].values;  N = df['N'].values
        E_rel = E - E[0];    N_rel = N - N[0]
        theta0 = np.arctan2(N_rel[1]-N_rel[0], E_rel[1]-E_rel[0])
        cos0, sin0 = np.cos(-theta0), np.sin(-theta0)
        E_rot =  E_rel * cos0 - N_rel * sin0
        N_rot =  E_rel * sin0 + N_rel * cos0


        # 4) pred 누적 궤적 계산
        x_pr = y_pr = hd_pr = 0.0
        tx_pr, ty_pr = [x_pr], [y_pr]
        for s, dh in zip(pred_speed, pred_hc):
            hd_pr += dh
            x_pr  += s * np.cos(hd_pr)
            y_pr  += s * np.sin(hd_pr)
            tx_pr.append(x_pr); ty_pr.append(y_pr)

        # 5-1) 궤적 비교 플롯
        plt.figure(figsize=(10, 8))
        plt.plot(E_rot, N_rot,      'b-',  label='GT Trajectory', linewidth=2)
        plt.plot(tx_pr, ty_pr,     'r-', label='Pred Trajectory', linewidth=2)
        plt.scatter([0], [0], c='green', s=100, label='Start')
        plt.title('GT vs Predicted Trajectory')
        plt.xlabel('Easting (m)'); plt.ylabel('Northing (m)')
        plt.legend(); plt.grid(); plt.axis('equal')
        plt.show()

        # 5-2) 속도 비교 플롯
        plt.figure(figsize=(10, 5))
        plt.plot(gt_speed_ws,     'b--',  label='GT Speed (downsampled)')
        plt.plot(pred_speed,      'r-', label='Pred Speed')
        plt.title('GT vs Predicted Speed')
        plt.xlabel('Window Index'); plt.ylabel('Speed (m/s)')
        plt.legend(); plt.grid(); plt.show()

        # 5-3) 헤딩 변화량 비교 플롯
        plt.figure(figsize=(10, 5))
        gt_hc_full = df['heading_diff'].values * 0.1
        gt_hc_ws   = gt_hc_full[stride - 1 :: stride][:len(pred_hc)]
        plt.plot(gt_hc_ws,      'b--',  label='GT Heading Change')
        plt.plot(pred_hc,       'r-', label='Pred Heading Change')
        plt.title('GT vs Predicted Heading Change')
        plt.xlabel('Window Index'); plt.ylabel('Heading Change (rad)')
        plt.legend(); plt.grid(); plt.show()