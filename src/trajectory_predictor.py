import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyproj import Proj
import math

class TrajectoryPredictor:
    """
    - 모델(model), sensor_scalers, y_speed_scaler, y_hc_scaler, window_size를 받아
      DataProcessor로 준비된 df에서 윈도우 생성 → 통계 피처 계산 → 스케일 → 예측 → 복원 → 시각화
    """

    def __init__(self, model, sensor_scalers, y_speed_scaler, y_hc_scaler, window_size):
        self.model = model
        self.sensor_scalers = sensor_scalers
        self.y_speed_scaler = y_speed_scaler
        self.y_hc_scaler = y_hc_scaler
        self.window_size = window_size

    @staticmethod
    def _apply_group_scale(win_2d: np.ndarray, scaler_entry, axes, eps=1e-8):
        """
        win_2d: (W, F) 윈도우
        scaler_entry:
        - (A) StandardScaler 객체  → .transform 사용
        - (B) dict {"mu":..., "sigma":..., "idxs":[...]} → 수식 적용
        axes: 그룹 축 인덱스 (예: [0,1,2])
        """
        if scaler_entry is None:
            return win_2d
        if hasattr(scaler_entry, "transform"):
            # (A) StandardScaler 같은 sklearn 객체
            win_2d[:, axes] = scaler_entry.transform(win_2d[:, axes])
        else:
            # (B) dict 저장된 그룹 파라미터
            mu = scaler_entry.get("mu", 0.0)
            sigma = scaler_entry.get("sigma", 1.0) + eps
            win_2d[:, axes] = (win_2d[:, axes] - mu) / sigma
        return win_2d
    @staticmethod
    def _apply_scalar_scale(vec_1d: np.ndarray, scaler_entry, eps=1e-8):
        """
        vec_1d: (W,) 혹은 (W,1)
        scaler_entry:
        - (A) StandardScaler 등 sklearn 객체 → .transform 사용
        - (B) dict {"mu":..., "sigma":...} → 수식 적용
        """
        if scaler_entry is None:
            return vec_1d
        if hasattr(scaler_entry, "transform"):
            # vec_1d shape 보정
            v = vec_1d.reshape(-1, 1)
            v = scaler_entry.transform(v)
            return v.ravel()
        else:
            mu = float(scaler_entry.get("mu", 0.0))
            sigma = float(scaler_entry.get("sigma", 1.0)) + eps
            return (vec_1d - mu) / sigma
    
    
    def _prepare_windows(self, df: pd.DataFrame, stride: int = 5) -> np.ndarray:
        cols = ['Accelerometer x','Accelerometer y','Accelerometer z',
                'Gyroscope x','Gyroscope y','Gyroscope z',
                'Acc_Norm', 'Gyro_Norm']
        arr = df[cols].values.astype(np.float32)

        M = len(df)
        W = self.window_size
        wins = []

        acc_axes  = [0,1,2]
        gyro_axes = [3,4,5]
        acc_norm_idx, gyro_norm_idx = 6, 7
        

        sc_acc   = self.sensor_scalers.get("acc_group", None)
        sc_gyro  = self.sensor_scalers.get("gyro_group", None)
        sc_anorm = self.sensor_scalers.get("acc_norm", None)   # ← Trainer에서 저장된 항목 사용
        sc_gnorm = self.sensor_scalers.get("gyro_norm", None)

        for start in range(0, M - W + 1, stride):
            win = arr[start:start+W].copy()  # (W, 8)

            # --- (1) acc/gyro 3축 그룹 스케일 ---
            win = self._apply_group_scale(win, sc_acc,  acc_axes)
            win = self._apply_group_scale(win, sc_gyro, gyro_axes)

            # --- (2) Norm은 '재계산'하지 않고, 기존 값에 스케일만 적용 ---
            if acc_norm_idx is not None:
                win[:, acc_norm_idx] = self._apply_scalar_scale(win[:, acc_norm_idx], sc_anorm)
            if gyro_norm_idx is not None:
                win[:, gyro_norm_idx] = self._apply_scalar_scale(win[:, gyro_norm_idx], sc_gnorm)

            wins.append(win)

        return np.array(wins, dtype=np.float32)


    def predict_and_plot_trajectory(self, df: pd.DataFrame, plag_1Hz: bool = False, title: str = "" ):
        """
        df에는 load_and_preprocess_csv 로 얻은 원본 50Hz DataFrame이 들어온다고 가정.
        1) plag_1Hz=False → stride=5, plag_1Hz=True → stride=50 로 윈도우 생성하여 예측 (스케일 복원 포함)
        2) 예측된 속도·헤딩을 plot
        3) 예측 궤적(accumulate) plot
        """

        stride = self.window_size if plag_1Hz else self.window_size // 10

        #stride = 1
        X = self._prepare_windows(df, stride)  # shape = (num_windows, window_size, num_features)

        # 1) 모델 예측 (스케일된 Y_pred_scaled)
        Y_pred_scaled = self.model.predict(X)  # shape = (num_windows, 2)

        # 2) 스케일 복원
        #    - 첫 번째 열은 속도, 두 번째 열은 헤딩 변화량
        pred_speed = self.y_speed_scaler.inverse_transform(Y_pred_scaled[:, 0].reshape(-1, 1)).ravel()
        pred_hc    = self.y_hc_scaler.inverse_transform(Y_pred_scaled[:, 1].reshape(-1, 1)).ravel()
        
        # noise 제거를 위해 10도 이하의 헤딩 변화량을 0으로 보정
        #pred_hc[np.abs(np.degrees(pred_hc)) < 10] = 0
        
        if not plag_1Hz:
            pred_speed = pred_speed * (stride/self.window_size)
            pred_hc    = pred_hc * (stride/self.window_size)
            
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
            
        # -------------------------------
        # Predicted Movement Trajectory plot
        # -------------------------------
        plt.figure(figsize=(8, 8))
        plt.plot(traj_x, traj_y, 'b-', alpha=0.9, markersize=3, label='Predicted Path')
        plt.scatter([traj_x[0]], [traj_y[0]], c='green', s=60, label='Start')
        plt.scatter([traj_x[-1]], [traj_y[-1]], c='red', s=60, label='End')
        plt.title(title)  
        plt.xlabel('East (m)')
        plt.ylabel('North (m)')
        plt.xlim(-25, 25)
        plt.ylim(-25, 25)
        plt.grid(True)
        plt.axis('equal')
        plt.legend()
        plt.show()
        
        # === ⬇️ 시작점-종료점 거리 계산 및 출력 추가 ===
        start_x, start_y = traj_x[0], traj_y[0]
        end_x, end_y = traj_x[-1], traj_y[-1]
        dist = np.hypot(end_x - start_x, end_y - start_y)
        print(f"Start-End distance: {dist:.3f} m")

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

        return np.vstack([pred_speed, pred_hc]).T, (traj_x, traj_y)

    def compare_trajectories(self, df: pd.DataFrame, plag_1Hz: bool = False):
        """
        1) df와 Y를 입력받아서, model.predict 결과와 GT(Y)를 비교
           - df: load_and_preprocess_csv로 얻은 DataFrame (50Hz 샘플링)
           - Y : shape=(n_windows, 2) 형태의 GT 값 ([speed_1, heading_1])이지만,
                 실제로 여기서는 df에 있는 컬럼 'speed_1', 'heading_1'을 사용하므로
                 길이가 50Hz(샘플 수) 만큼 존재한다 가정
        2) plag_1Hz=False → stride=5, plag_1Hz=True → stride=50 로 윈도우 생성
        3) pred_speed, pred_hc 예측 → 스케일 복원
        4) GT 속도·헤딩 vs 예측 속도·헤딩 비교 (길이 맞춤)
        5) GT Trajectory vs Pred Trajectory 비교
        """

        # ===== 1) stride 결정 및 윈도우 시작 인덱스 목록 생성 =====
        stride = self.window_size if plag_1Hz else self.window_size // 10
        window_size = self.window_size

        # ===== 2) 윈도우 단위 예측 =====
        X = self._prepare_windows(df, stride)   # (num_windows, window_size, num_features)
        Y_pred_scaled = self.model.predict(X)           # (num_windows, 2)

        # 스케일 복원
        pred_speed = self.y_speed_scaler.inverse_transform(
            Y_pred_scaled[:, 0].reshape(-1, 1)
        ).ravel()
        pred_hc = self.y_hc_scaler.inverse_transform(
            Y_pred_scaled[:, 1].reshape(-1, 1)
        ).ravel()

        if not plag_1Hz:
            pred_speed = pred_speed * (stride/window_size)
            pred_hc    = pred_hc * (stride/window_size) 
            
        plt.plot(np.degrees(pred_hc), '.-')
        plt.show()
        
        
        M = len(df)                     # 예: 50Hz 데이터 개수
        n_windows_Hz1 = M // 50         # 초당 1개 대표 좌표 개수

        df_ori_e = []
        df_ori_n = []
        for i in range(n_windows_Hz1):
            center_idx = i * 50 + 25    # 가운데 인덱스 (0~49 → 25, 50~99 → 75, ...)
            if center_idx < M:
                df_ori_e.append(df['E'].iloc[center_idx])
                df_ori_n.append(df['N'].iloc[center_idx])
            else:
                # 만약 마지막 윈도우가 50미만 남았으면 마지막 샘플 사용
                df_ori_e.append(df['E'].iloc[-1])
                df_ori_n.append(df['N'].iloc[-1])

        # N: 1Hz 대표 좌표 개수
        N = len(df_ori_e)
        
        
        E = df_ori_e
        N = df_ori_n
        dx = E[1] - E[0]
        dy = N[1] - N[0]
        
        theta = math.atan2(dy, dx)
        cos_a = math.cos(-theta)
        sin_a = math.sin(-theta)
        R = np.array([[cos_a, -sin_a],
                    [sin_a,  cos_a]])

        # 4) 원점(초기점)을 빼서 상대좌표로 만든 뒤 회전
        coords = np.vstack((
            E - E[0],    # East 방향 상대좌표
            N - N[0]     # North 방향 상대좌표
        ))
        rotated = R @ coords

        e_corr = rotated[0, :]
        n_corr = rotated[1, :]


        # -------------------------------
        # 7) Pred Trajectory 누적 적분
        # -------------------------------
        hd_pr = 0.0
        x_pr = 0.0
        y_pr = 0.0
        tx_pr, ty_pr = [x_pr], [y_pr]
        for s, dh in zip(pred_speed, pred_hc):
            hd_pr += dh 
            x_pr += s * np.cos(hd_pr)
            y_pr += s * np.sin(hd_pr)
            tx_pr.append(x_pr)
            ty_pr.append(y_pr)

        # -------------------------------
        # 8) GT vs Pred Trajectory 비교 플롯
        # -------------------------------
        plt.figure(figsize=(10, 8))
        plt.plot(e_corr,        n_corr,   'b-', label='GT Trajectory',   linewidth=2)
        plt.plot(tx_pr,        ty_pr,   'r-', label='Pred Trajectory', linewidth=2)
        plt.scatter([0], [0], c='green', s=100, label='Start')
        plt.title(f'GT vs Predicted Trajectory (stride={stride})')
        plt.xlabel('Easting (m)')
        plt.ylabel('Northing (m)')
        plt.legend()
        plt.grid()
        plt.axis('equal')
        plt.show()
