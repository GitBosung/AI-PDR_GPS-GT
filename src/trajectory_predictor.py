# src/trajectory_predictor.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class TrajectoryPredictor:
    """
    학습된 모델을 사용하여 새로운 센서 데이터에 대해 예측 및 이동 경로 시각화를 수행하는 클래스
      - 학습 시 사용한 scaler 적용
      - 슬라이딩 윈도우 방식으로 입력 데이터를 생성하여 예측 수행
      - 예측 결과를 기반으로 실제 경로와 예측 경로 비교
    """
    def __init__(self, model, scaler, window_size=50):
        self.model = model
        self.scaler = scaler
        self.window_size = window_size

    def _prepare_sensor_data(self, df):
        """
        예측에 사용할 센서 데이터를 준비함.
          - 사용 피처: 10개
            [Accelerometer x, Accelerometer y, Accelerometer z,
             Gyroscope x, Gyroscope y, Gyroscope z,
             Acc_Norm]
          - 전체 피처에 대해 scaler 적용
        """
        sensor_columns = ['Accelerometer x', 'Accelerometer y', 'Accelerometer z',
                          'Gyroscope x', 'Gyroscope y', 'Gyroscope z',
                          'Acc_Norm']
        
        sensor_data = df[sensor_columns].copy().astype('float32')
        scaled_array = self.scaler.transform(sensor_data.values)
        sensor_data_scaled = pd.DataFrame(scaled_array, columns=sensor_columns)
        
        return sensor_data_scaled

    def predict_and_plot_trajectory(self, df):
        """
        새로운 테스트 데이터에 대해 예측을 수행한 후 이동 경로를 시각화
        """
        sensor_data = self._prepare_sensor_data(df)
        
        # 슬라이딩 윈도우 방식 입력 데이터 생성
        X_test_new = []
        for i in range(0, len(sensor_data) - self.window_size, self.window_size):
            window = sensor_data.iloc[i:i + self.window_size].values
            X_test_new.append(window)
        X_test_new = np.array(X_test_new).astype('float32')
        
        # 모델 예측 (출력: [속도, 헤딩 변화량])
        Y_pred = self.model.predict(X_test_new)
        
        # 예측 결과로부터 이동 경로 계산 (누적 방식)
        x, y = 0, 0
        trajectory_x, trajectory_y = [x], [y]
        heading = 0
        for speed, heading_change in Y_pred:
            heading += heading_change
            dx = speed * np.cos(heading)
            dy = speed * np.sin(heading)
            x += dx
            y += dy
            trajectory_x.append(x)
            trajectory_y.append(y)
        
        plt.figure(figsize=(8, 6))
        plt.plot(trajectory_x, trajectory_y, 'b-', alpha=0.7, label='Trajectory')
        plt.plot(trajectory_x, trajectory_y, 'bo', markersize=3, alpha=0.5)
        plt.plot(trajectory_x[0], trajectory_y[0], 'ro', markersize=8, label="Start")
        plt.plot(trajectory_x[-1], trajectory_y[-1], 'go', markersize=8, label="End")
        plt.xlabel('Easting (m)')
        plt.ylabel('Northing (m)')
        plt.title('Predicted Movement Trajectory')
        plt.legend()
        plt.grid()
        plt.axis('equal')
        plt.show()

    def compare_trajectories(self, df):
        """
        Ground Truth 경로와 예측 경로를 비교하고, 속도 및 헤딩 변화량 비교 플롯을 생성
        """
        # Ground Truth: 헤딩 변화량이 있는 유효 인덱스 선택
        valid_indices = df['Heading Change'].notna()
        gt_speed = df['Speed'][valid_indices].values
        gt_heading_change = np.unwrap(df['Heading Change'][valid_indices].values)
        
        sensor_data = self._prepare_sensor_data(df)
        
        X_test_new = []
        for i in range(0, len(sensor_data) - self.window_size, self.window_size):
            window = sensor_data.iloc[i:i + self.window_size].values
            X_test_new.append(window)
        X_test_new = np.array(X_test_new).astype('float32')
        
        Y_pred = self.model.predict(X_test_new)
        pred_speed = Y_pred[:, 0]
        pred_heading_change = Y_pred[:, 1]
        
        # Ground Truth 이동 경로 계산
        x_gt, y_gt = 0, 0
        trajectory_x_gt, trajectory_y_gt = [x_gt], [y_gt]
        heading_gt = 0
        for speed, heading_change in zip(gt_speed, gt_heading_change):
            heading_gt += heading_change
            dx = speed * np.cos(heading_gt)
            dy = speed * np.sin(heading_gt)
            x_gt += dx
            y_gt += dy
            trajectory_x_gt.append(x_gt)
            trajectory_y_gt.append(y_gt)
        
        # 예측 이동 경로 계산
        x_pred, y_pred = 0, 0
        trajectory_x_pred, trajectory_y_pred = [x_pred], [y_pred]
        heading_pred = 0
        for speed, heading_change in zip(pred_speed, pred_heading_change):
            heading_pred += heading_change
            dx = speed * np.cos(heading_pred)
            dy = speed * np.sin(heading_pred)
            x_pred += dx
            y_pred += dy
            trajectory_x_pred.append(x_pred)
            trajectory_y_pred.append(y_pred)
        
        plt.figure(figsize=(10, 8))
        plt.plot(trajectory_x_gt, trajectory_y_gt, 'b-', label='Ground Truth', alpha=0.7)
        plt.plot(trajectory_x_gt, trajectory_y_gt, 'bo', markersize=3, alpha=0.5)
        plt.plot(trajectory_x_pred, trajectory_y_pred, 'r-', label='Predicted Path', alpha=0.7)
        plt.plot(trajectory_x_pred, trajectory_y_pred, 'ro', markersize=3, alpha=0.5)
        plt.plot(trajectory_x_gt[0], trajectory_y_gt[0], 'go', markersize=8, label='Start Point')
        plt.plot(trajectory_x_gt[-1], trajectory_y_gt[-1], 'bo', markersize=8, label='Ground Truth End')
        plt.plot(trajectory_x_pred[-1], trajectory_y_pred[-1], 'ro', markersize=8, label='Predicted End')
        plt.xlabel('Easting (m)')
        plt.ylabel('Northing (m)')
        plt.title('Ground Truth vs Predicted Trajectory')
        plt.legend()
        plt.grid()
        plt.axis('equal')
        plt.show()
        
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        axs[0, 0].plot(gt_speed, 'b-', label='Ground Truth')
        axs[0, 0].plot(pred_speed, 'r-', label='Predicted')
        axs[0, 0].set_title('Speed Comparison')
        axs[0, 0].set_xlabel('Time')
        axs[0, 0].set_ylabel('Speed (m/s)')
        axs[0, 0].legend()
        axs[0, 0].grid()
        
        axs[0, 1].plot(np.degrees(gt_heading_change), 'b-', label='Ground Truth')
        axs[0, 1].plot(np.degrees(pred_heading_change), 'r-', label='Predicted')
        axs[0, 1].set_title('Heading Change Comparison')
        axs[0, 1].set_xlabel('Time')
        axs[0, 1].set_ylabel('Heading Change (degrees)')
        axs[0, 1].legend()
        axs[0, 1].grid()
        
        speed_error = np.abs(gt_speed - pred_speed[:len(gt_speed)])
        axs[1, 0].plot(speed_error, 'g-')
        axs[1, 0].set_title('Speed Error')
        axs[1, 0].set_xlabel('Time')
        axs[1, 0].set_ylabel('Speed Error (m/s)')
        axs[1, 0].grid()
        
        heading_error = np.abs(np.degrees(gt_heading_change - pred_heading_change[:len(gt_heading_change)]))
        axs[1, 1].plot(heading_error, 'g-')
        axs[1, 1].set_title('Heading Change Error')
        axs[1, 1].set_xlabel('Time')
        axs[1, 1].set_ylabel('Heading Change Error (degrees)')
        axs[1, 1].grid()
        
        plt.tight_layout()
        plt.show()
        
        mean_speed_error = np.mean(speed_error)
        mean_heading_error = np.mean(heading_error)
        print(f"Mean Speed Error: {mean_speed_error:.4f} m/s")
        print(f"Mean Heading Change Error: {mean_heading_error:.4f} degrees")
