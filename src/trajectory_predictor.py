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
            'Acc_Norm', #'Gyro_Norm'
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


    def predict_and_plot_trajectory(self, df: pd.DataFrame, plag_1Hz: bool = False, title: str = "" , plot_flag: bool = False):
        """
        df에는 load_and_preprocess_csv 로 얻은 원본 50Hz DataFrame이 들어온다고 가정.
        1) plag_1Hz=False → stride=5, plag_1Hz=True → stride=50 로 윈도우 생성하여 예측 (스케일 복원 포함)
        2) 예측된 속도·헤딩을 plot
        3) 예측 궤적(accumulate) plot
        
        pram df: DataFrame
        pram plag_1Hz: True → 1Hz 예측, False → 10Hz 예측, default=False
        pram title: plot 제목, default=""
        pram plot_flag: True → 궤적 플롯 시 좌표 반전 (우회전인 경우), default=False
        return: 시작점-종료점 거리 (float)
        """

        stride = self.window_size if plag_1Hz else self.window_size // 10

        #stride = 1
        X = self._prepare_windows(df, stride)  # shape = (num_windows, window_size, num_features)

        Y_pred_scaled = self.model.predict(X)

        # ✅ 2-head output을 (N,2)로 합치기
        if isinstance(Y_pred_scaled, dict):
            y_speed_s = Y_pred_scaled["speed"]
            y_dh_s    = Y_pred_scaled["dh"]
            Y_pred_scaled = np.hstack([y_speed_s, y_dh_s])
        elif isinstance(Y_pred_scaled, (list, tuple)):
            y_speed_s, y_dh_s = Y_pred_scaled
            Y_pred_scaled = np.hstack([y_speed_s, y_dh_s])

        # ✅ y_scaler 1개로 inverse
        Y_pred = self.y_scaler.inverse_transform(Y_pred_scaled)
        pred_speed = Y_pred[:, 0]
        pred_hc    = Y_pred[:, 1]
        
        # 10Hz 예측인 경우, 윈도우 간격에 맞게 속도·헤딩 변화량 보정
        if not plag_1Hz:
            pred_speed = pred_speed * (stride/self.window_size)
            pred_hc    = pred_hc * (stride/self.window_size)
            
        #-2 ~ 2 degree 구간이면 0으로 처리
        #pred_hc = np.where(np.abs(np.degrees(pred_hc)) <= 1.0, 0, pred_hc)
            
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
            
        if plot_flag:
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
        
        return plot_x, plot_y
    
    def analyze_pdr_performance(self,gt_df, pred_x, pred_y, title="PDR Performance Analysis"):
        """
        GT 데이터와 예측 데이터를 비교하여 Plot을 그리고 정량적 지표(RMSE, MAE)를 산출합니다.
        gt_df: 이전에 생성한 gt_synced_stride20_*.csv 기반 DataFrame (N, 2)
        pred_x, pred_y: 모델 함수에서 리턴받은 예측 궤적 (N, 2)
        """
        # 1. 길이 동기화 확인 (예측치가 1개 더 많을 수 있으므로 슬라이싱)
        min_len = min(len(gt_df), len(pred_x))
        gt_coords = gt_df[['GT_X', 'GT_Y']].values[:min_len]
        pred_coords = np.column_stack((pred_x[:min_len], pred_y[:min_len]))

        # 2. 정량적 지표 계산
        errors = np.sqrt(np.sum((gt_coords - pred_coords)**2, axis=1))
        rmse = np.sqrt(np.mean(errors**2))
        mae = np.mean(errors)
        max_error = np.max(errors)
        
        # 시작점-종료점 거리 오차 (Return Position Error)
        rpe_gt = np.hypot(gt_coords[-1, 0] - gt_coords[0, 0], gt_coords[-1, 1] - gt_coords[0, 1])
        rpe_pred = np.hypot(pred_coords[-1, 0] - pred_coords[0, 0], pred_coords[-1, 1] - pred_coords[0, 1])
        loop_closure_error = np.hypot(pred_coords[-1, 0] - gt_coords[-1, 0], pred_coords[-1, 1] - gt_coords[-1, 1])

        # 3. 결과 출력
        print(f"\n===== {title} Statistics =====")
        print(f"RMSE: {rmse:.4f} m")
        print(f"MAE: {mae:.4f} m")
        print(f"Max Error: {max_error:.4f} m")
        print(f"Final Point Error (Loop Closure): {loop_closure_error:.4f} m")
        print("==========================================\n")

        # 4. Trajectory 비교 Plot
        plt.figure(figsize=(10, 10))
        plt.plot(gt_coords[:, 0], gt_coords[:, 1], 'k--', label='Ground Truth (DCM)', alpha=0.8)
        plt.plot(pred_coords[:, 0], pred_coords[:, 1], 'b-o', label='AI-PDR Prediction', markersize=3, alpha=0.7)
        
        plt.scatter(gt_coords[0, 0], gt_coords[0, 1], c='green', s=100, label='Start', zorder=5)
        plt.scatter(gt_coords[-1, 0], gt_coords[-1, 1], c='red', marker='x', s=100, label='GT End', zorder=5)
        plt.scatter(pred_coords[-1, 0], pred_coords[-1, 1], c='magenta', s=100, label='Predict End', zorder=5)
        
        plt.title(f"{title}\nRMSE: {rmse:.3f}m, MAE: {mae:.3f}m")
        plt.xlabel('East (m)')
        plt.ylabel('North (m)')
        plt.axis('equal')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend()
        plt.show()

        # 5. Error Distribution (CDF) Plot - IEEE 저널 필수 항목
        sorted_errors = np.sort(errors)
        cdf = np.arange(len(sorted_errors)) / float(len(sorted_errors))
        
        plt.figure(figsize=(8, 5))
        plt.plot(sorted_errors, cdf, linewidth=2, color='red')
        plt.title('Cumulative Distribution Function of Error')
        plt.xlabel('Error (meters)')
        plt.ylabel('Probability')
        plt.grid(True)
        plt.show()

        return rmse, mae

 



