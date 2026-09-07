import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyproj import Proj
import math
from matplotlib.animation import FuncAnimation

class TrajectoryPredictor:
    """
    - 모델(model), sensor_scalers, y_speed_scaler, y_hc_scaler, window_size를 받아
      DataProcessor로 준비된 df에서 윈도우 생성 → 통계 피처 계산 → 스케일 → 예측 → 복원 → 시각화
    """

    def __init__(self, model, x_scaler, y_scaler, window_size):
        self.model = model
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.window_size = window_size
    
    def _prepare_windows(self, df: pd.DataFrame, stride: int = 5) -> np.ndarray:
        cols = [
            'Accelerometer x','Accelerometer y','Accelerometer z',
            'Gyroscope x','Gyroscope y','Gyroscope z',
            'Acc_Norm'
        ]
        
        arr = df[cols].values.astype(np.float32)

        M = len(arr)
        W = self.window_size
        wins = []

        for start in range(0, M - W + 1, stride):
            win = arr[start:start+W].copy()   # (W, F)

            # ---- scaler 적용 (train과 동일 방식) ----
            F = win.shape[1]
            win_2d = win.reshape(-1, F)
            win_2d = self.x_scaler.transform(win_2d)
            win = win_2d.reshape(W, F)

            wins.append(win)

        return np.array(wins, dtype=np.float32)


    def predict_and_plot_trajectory(self, df: pd.DataFrame, rotation_flag: bool = False, title: str = "Predicted Trajectory"):
        stride = 20
        X = self._prepare_windows(df, stride)  # shape = (num_windows, window_size, num_features)

        Y_pred_scaled = self.model.predict(X)

        if isinstance(Y_pred_scaled, dict):
            y_speed_s = Y_pred_scaled["speed"]
            y_dh_s    = Y_pred_scaled["dh"]
            Y_pred_scaled = np.hstack([y_speed_s, y_dh_s])
        elif isinstance(Y_pred_scaled, (list, tuple)):
            y_speed_s, y_dh_s = Y_pred_scaled
            Y_pred_scaled = np.hstack([y_speed_s, y_dh_s])

        Y_pred = self.y_scaler.inverse_transform(Y_pred_scaled)
        pred_speed = Y_pred[:, 0]
        pred_hc    = Y_pred[:, 1]
        
        pred_speed = pred_speed * (stride/self.window_size)
        pred_hc    = pred_hc * (stride/self.window_size)
        
        send_heading = []
            
        # -------------------------------
        # 속도 예측 그래프
        # -------------------------------
        plt.figure(figsize=(8, 4))
        plt.plot(pred_speed, '-o', label='Predicted Speed (m/s)', markersize=4)
        plt.title(f'Predicted Speed (stride={stride})')
        plt.xlabel('Window Index')
        plt.ylabel('Speed (m/s)')
        plt.grid(True)
        plt.legend()
        plt.show()

        # -------------------------------
        # 헤딩 변화 예측 그래프 (deg 단위)
        # -------------------------------
        plt.figure(figsize=(8, 4))
        plt.plot(np.degrees(pred_hc), '-o', label='Predicted Heading Change (deg)', markersize=4)
        plt.title(f'Predicted Heading Change (stride={stride})')
        plt.xlabel('Window Index')
        plt.ylabel('Heading Change (deg)')
        plt.grid(True)
        plt.legend()
        plt.show()
        
        

        # -------------------------------
        # Predicted 궤적 누적 적분
        # -------------------------------
        arr_heading = [0]
        x = y = heading = 0.0
        traj_x, traj_y = [x], [y]
        for s, dh in zip(pred_speed, pred_hc):
            heading += dh
            arr_heading.append(heading)
            x += s * np.cos(heading)
            y += s * np.sin(heading)
            traj_x.append(x)
            traj_y.append(y)
            
        if rotation_flag:
            plot_x = -np.array(traj_x)
            plot_y = -np.array(traj_y)
        else:
            plot_x = np.array(traj_x)
            plot_y = np.array(traj_y)
            
        # -------------------------------
        # Predicted Movement Trajectory plot
        # -------------------------------
        plt.figure(figsize=(8, 8))
        plt.plot(plot_x, plot_y, 'b-', alpha=0.9, markersize=3, label='Predicted Path')
        plt.scatter([plot_x[0]], [plot_y[0]], c='green', s=60, label='Start')
        plt.scatter([plot_x[-1]], [plot_y[-1]], c='red', s=60, label='End')
        plt.title(title)  
        plt.xlabel('East (m)')
        plt.ylabel('North (m)')
        plt.xlim(-25, 25)
        plt.ylim(-25, 25)
        plt.grid(True)
        plt.axis('equal')
        plt.legend()
        plt.show()
        
        plt.plot(np.degrees(arr_heading), 'r-', label='Heading (deg)')
        plt.title('Cumulative Heading Change')
        plt.xlabel('Time (s)')
        plt.ylabel('Heading (deg)')
        plt.legend()
        plt.grid(True, which='both', axis='y')
        plt.yticks(np.arange(
            int(np.floor(np.min(np.degrees(arr_heading)) / 90) * 90),
            int(np.ceil(np.max(np.degrees(arr_heading)) / 90) * 90) + 1,
            90
        ))
        plt.show()
        
        return plot_x, plot_y, arr_heading
    