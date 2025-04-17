import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class TrajectoryPredictor:
    """
    - 학습된 모델과 sensor_scalers(dict)를 받아,
      축별로 스케일링한 뒤
      슬라이딩 윈도우 기반 예측 & 시각화를 수행
    """
    def __init__(self, model, sensor_scalers: dict, window_size=50):
        self.model = model
        self.sensor_scalers = sensor_scalers
        self.window_size = window_size

    def _prepare_sensor_data(self, df: pd.DataFrame) -> pd.DataFrame:
        sensor_cols = [
            'Accelerometer x', 'Accelerometer y', 'Accelerometer z',
            'Gyroscope x', 'Gyroscope y', 'Gyroscope z',
            'Acc_Norm'
        ]
        raw = df[sensor_cols].astype(np.float32).copy()
        # 결과를 담을 numpy array
        arr = raw.values  # shape = (n_samples, 7)
        scaled_arr = np.zeros_like(arr, dtype=np.float32)

        # 축별 스케일러 적용
        for idx, col in enumerate(sensor_cols):
            scaler = self.sensor_scalers.get(col)
            if scaler is None:
                # 스케일러가 없으면 원본 그대로
                scaled_arr[:, idx] = arr[:, idx]
            else:
                # sklearn transform은 2D array 입력 필요
                col_data = arr[:, idx].reshape(-1, 1)
                scaled_arr[:, idx] = scaler.transform(col_data).ravel()

        # DataFrame으로 재구성
        sensor_scaled = pd.DataFrame(scaled_arr, columns=sensor_cols)
        return sensor_scaled

    def predict_and_plot_trajectory(self, df):
        sensor_data = self._prepare_sensor_data(df)
        X_test_new = [
            sensor_data.iloc[i:i + self.window_size].values
            for i in range(0, len(sensor_data) - self.window_size, self.window_size)
        ]
        X_test_new = np.array(X_test_new, dtype=np.float32)

        Y_pred = self.model.predict(X_test_new)

        # 누적 궤적 계산
        x = y = heading = 0.0
        traj_x, traj_y = [x], [y]
        for speed, dh in Y_pred:
            heading += dh
            x += speed * np.cos(heading)
            y += speed * np.sin(heading)
            traj_x.append(x)
            traj_y.append(y)

        plt.figure(figsize=(8, 6))
        plt.plot(traj_x, traj_y, 'b-o', alpha=0.7, label='Predicted Path', markersize=4)
        plt.plot(traj_x[0], traj_y[0], 'go', markersize=8, label='Start')
        plt.plot(traj_x[-1], traj_y[-1], 'ro', markersize=8, label='End')
        plt.xlabel('Easting (m)')
        plt.ylabel('Northing (m)')
        plt.title('Predicted Movement Trajectory')
        plt.legend()
        plt.grid()
        plt.axis('equal')
        plt.show()

    def compare_trajectories(self, df):
        # Ground truth 속도/헤딩
        gt_mask_s = df['Pure Speed'] != 0
        gt_speed = df.loc[gt_mask_s, 'Pure Speed'].values
        gt_mask_h = df['Pure Heading Change'] != 0
        gt_hc = np.unwrap(df.loc[gt_mask_h, 'Pure Heading Change'].values)

        # 예측
        sensor_data = self._prepare_sensor_data(df)
        X_test_new = [
            sensor_data.iloc[i:i + self.window_size].values
            for i in range(0, len(sensor_data) - self.window_size, self.window_size)
        ]
        X_test_new = np.array(X_test_new, dtype=np.float32)
        Y_pred = self.model.predict(X_test_new)
        pred_speed, pred_hc = Y_pred[:,0], Y_pred[:,1]

        # GT 궤적
        x_gt = y_gt = hd_gt = 0.0
        tx_gt, ty_gt = [x_gt], [y_gt]
        for s, dh in zip(gt_speed, gt_hc):
            hd_gt += dh
            x_gt += s * np.cos(hd_gt)
            y_gt += s * np.sin(hd_gt)
            tx_gt.append(x_gt); ty_gt.append(y_gt)

        # Pred 궤적
        x_pr = y_pr = hd_pr = 0.0
        tx_pr, ty_pr = [x_pr], [y_pr]
        for s, dh in zip(pred_speed, pred_hc):
            hd_pr += dh
            x_pr += s * np.cos(hd_pr)
            y_pr += s * np.sin(hd_pr)
            tx_pr.append(x_pr); ty_pr.append(y_pr)

        # 궤적 비교
        plt.figure(figsize=(10,8))
        plt.plot(tx_gt, ty_gt, 'b-o', label='GT', alpha=0.7, markersize=4)
        plt.plot(tx_pr, ty_pr, 'r-o', label='Pred', alpha=0.7, markersize=4)
        plt.plot(tx_gt[0], ty_gt[0], 'go', markersize=8, label='Start')
        plt.plot(tx_gt[-1], ty_gt[-1], 'bo', markersize=8, label='GT End')
        plt.plot(tx_pr[-1], ty_pr[-1], 'ro', markersize=8, label='Pred End')
        plt.xlabel('Easting (m)'); plt.ylabel('Northing (m)')
        plt.title('GT vs Predicted Trajectory'); plt.legend(); plt.grid(); plt.axis('equal')
        plt.show()

        # 속도·헤딩 에러
        sp_err = np.abs(gt_speed - pred_speed[:len(gt_speed)])
        hd_err = np.abs(np.degrees(gt_hc - pred_hc[:len(gt_hc)]))

        fig, axs = plt.subplots(2,2, figsize=(15,10))
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

        plt.tight_layout()
        plt.show()
