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
        # 1) GT 궤적
        gt_mask_s = df['Pure Speed'] != 0
        gt_speed  = df.loc[gt_mask_s, 'Pure Speed'].values
        gt_mask_h = df['Pure Heading Change'] != 0
        gt_hc     = np.unwrap(df.loc[gt_mask_h, 'Pure Heading Change'].values)

        if plag_1Hz:
            X = self._prepare_windows_1Hz(df)               # (num_windows, window_size, num_features)
        else:
            X = self._prepare_windows(df)    
            # (num_windows, window_size, num_features)
        Y = self.model.predict(X) 
        
        if plag_1Hz:
            pred_speed = Y[:, 0] 
            pred_hc    = Y[:, 1] 
        else:
            pred_speed = Y[:, 0] * 0.1
            pred_hc    = Y[:, 1] * 0.1

        # 3) GT 누적 궤적
        x_gt = y_gt = hd_gt = 0.0
        tx_gt, ty_gt = [x_gt], [y_gt]
        for s, dh in zip(gt_speed, gt_hc):
            hd_gt += dh
            x_gt += s * np.cos(hd_gt)
            y_gt += s * np.sin(hd_gt)
            tx_gt.append(x_gt); ty_gt.append(y_gt)

        # 4) Pred 누적 궤적
        x_pr = y_pr = hd_pr = 0.0
        tx_pr, ty_pr = [x_pr], [y_pr]
        for s, dh in zip(pred_speed, pred_hc):
            hd_pr += dh
            x_pr += s * np.cos(hd_pr)
            y_pr += s * np.sin(hd_pr)
            tx_pr.append(x_pr); ty_pr.append(y_pr)

        # 5) 궤적 비교 시각화
        plt.figure(figsize=(10,8))
        plt.plot(tx_gt, ty_gt, 'b-o', label='GT', alpha=0.7, markersize=4)
        plt.plot(tx_pr, ty_pr, 'r-o', label='Pred', alpha=0.7, markersize=4)
        plt.plot(tx_gt[0], ty_gt[0], 'go', markersize=8, label='Start')
        plt.plot(tx_gt[-1], ty_gt[-1], 'bo', markersize=8, label='GT End')
        plt.plot(tx_pr[-1], ty_pr[-1], 'ro', markersize=8, label='Pred End')
        plt.title('GT vs Predicted Trajectory')
        plt.xlabel('Easting (m)'); plt.ylabel('Northing (m)')
        plt.legend(); plt.grid(); plt.axis('equal'); plt.show()

        # 6) 속도·헤딩 오차 플롯
        sp_err = np.abs(gt_speed - pred_speed[:len(gt_speed)])
        hd_err = np.abs(np.degrees(gt_hc - pred_hc[:len(gt_hc)]))

        fig, axs = plt.subplots(2,2, figsize=(12,10))
        axs[0,0].plot(gt_speed, '-o', label='GT', markersize=3)
        axs[0,0].plot(pred_speed, '-o', label='Pred', markersize=3)
        axs[0,0].set_title('Speed Comparison'); axs[0,0].legend(); axs[0,0].grid()

        axs[0,1].plot(np.degrees(gt_hc), '-o', label='GT', markersize=3)
        axs[0,1].plot(np.degrees(pred_hc), '-o', label='Pred', markersize=3)
        axs[0,1].set_title('Heading Change Comparison'); axs[0,1].legend(); axs[0,1].grid()

        axs[1,0].plot(sp_err, '-o', markersize=3)
        axs[1,0].set_title('Speed Error'); axs[1,0].grid()
        axs[1,1].plot(hd_err, '-o', markersize=3)
        axs[1,1].set_title('Heading Error (deg)'); axs[1,1].grid()

        plt.tight_layout(); plt.axis('equal'); plt.show()

        return (tx_gt, ty_gt), (tx_pr, ty_pr), sp_err, hd_err
