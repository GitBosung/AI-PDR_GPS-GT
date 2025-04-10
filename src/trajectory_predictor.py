import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class TrajectoryPredictor:
    """
    A class to perform trajectory prediction and visualization using the trained model.
    
    Main functionalities:
      - Extract and scale sensor data using the scalers from training.
      - Generate input data with a sliding window approach.
      - Compute and visualize predicted trajectory, along with speed and heading change distributions.
    """
    def __init__(self, model, scaler_acc, scaler_gyro, window_size=50):
        self.model = model
        self.scaler_acc = scaler_acc
        self.scaler_gyro = scaler_gyro
        self.window_size = window_size

    def _prepare_sensor_data(self, df, sensor_columns):
        """
        Extract sensor data from the DataFrame and apply scaling.
        """
        sensor_data = df[sensor_columns].copy().astype(np.float32)
        # Accelerometer scaling
        acc_data = sensor_data.iloc[:, 0:3].values
        acc_scaled = self.scaler_acc.transform(acc_data)
        sensor_data.iloc[:, 0:3] = acc_scaled
        # Gyroscope scaling
        gyro_data = sensor_data.iloc[:, 3:6].values
        gyro_scaled = self.scaler_gyro.transform(gyro_data)
        sensor_data.iloc[:, 3:6] = gyro_scaled

        return sensor_data

    def predict_and_plot_trajectory(self, df):
        """
        Predict trajectory using the model and visualize the trajectory and distributions.
        """
        sensor_columns = ['Accelerometer x', 'Accelerometer y', 'Accelerometer z',
                          'Gyroscope x', 'Gyroscope y', 'Gyroscope z']
        
        sensor_data = self._prepare_sensor_data(df, sensor_columns)
        X_test_new = []
        # 윈도우가 겹치지 않도록 window_size 간격으로 데이터를 자름
        for i in range(0, len(sensor_data), self.window_size):
            if i + self.window_size <= len(sensor_data):
                window = sensor_data.iloc[i:i + self.window_size].values
                X_test_new.append(window)
        X_test_new = np.array(X_test_new).astype(np.float32)
        Y_pred = self.model.predict(X_test_new)
        
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
        
        speed_values = Y_pred[:, 0]
        heading_change_values = Y_pred[:, 1]
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        axs[0].hist(speed_values, bins=50, edgecolor='black')
        axs[0].set_title('Speed Distribution')
        axs[0].set_xlabel('Speed')
        axs[0].set_ylabel('Frequency')
        axs[1].hist(np.degrees(heading_change_values), bins=50, edgecolor='black')
        axs[1].set_title('Heading Change Distribution')
        axs[1].set_xlabel('Heading Change (degrees)')
        axs[1].set_ylabel('Frequency')
        plt.tight_layout()
        plt.show()

    def compare_trajectories(self, df):
        """
        Compare ground truth and predicted trajectories.
        """
        valid_indices = df['Heading Change'].notna()
        gt_speed = df['Speed'][valid_indices].values
        gt_heading_change = df['Heading Change'][valid_indices].values
        gt_heading_change = np.unwrap(gt_heading_change)
        
        sensor_columns = ['Accelerometer x', 'Accelerometer y', 'Accelerometer z',
                          'Gyroscope x', 'Gyroscope y', 'Gyroscope z']
        sensor_data = self._prepare_sensor_data(df, sensor_columns)
        X_test_new = []
        for i in range(0, len(sensor_data), self.window_size):
            if i + self.window_size <= len(sensor_data):
                window = sensor_data.iloc[i:i + self.window_size].values
                X_test_new.append(window)
        X_test_new = np.array(X_test_new).astype(np.float32)
        
        # 모델 예측 수행
        Y_pred = self.model.predict(X_test_new)
        pred_speed = Y_pred[:, 0]
        pred_heading_change = Y_pred[:, 1]
        
        # Ground Truth 경로 계산
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
        
        # 예측 경로 계산
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
        
        # Ground Truth와 예측 경로 비교 시각화
        plt.figure(figsize=(10, 8))
        # Ground Truth 경로 (점과 선)
        plt.plot(trajectory_x_gt, trajectory_y_gt, 'b-', label='Ground Truth', alpha=0.7)
        plt.plot(trajectory_x_gt, trajectory_y_gt, 'bo', markersize=3, alpha=0.5)
        # 예측 경로 (점과 선)
        plt.plot(trajectory_x_pred, trajectory_y_pred, 'r-', label='Predicted Path', alpha=0.7)
        plt.plot(trajectory_x_pred, trajectory_y_pred, 'ro', markersize=3, alpha=0.5)
        # 시작점과 끝점 강조
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
        
        # 속도와 헤딩 변화량 비교
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # 속도 비교
        axs[0, 0].plot(gt_speed, 'b-', label='Ground Truth')
        axs[0, 0].plot(pred_speed, 'r-', label='Predicted')
        axs[0, 0].set_title('Speed Comparison')
        axs[0, 0].set_xlabel('Time')
        axs[0, 0].set_ylabel('Speed (m/s)')
        axs[0, 0].legend()
        axs[0, 0].grid()
        
        # 헤딩 변화량 비교
        axs[0, 1].plot(np.degrees(gt_heading_change), 'b-', label='Ground Truth')
        axs[0, 1].plot(np.degrees(pred_heading_change), 'r-', label='Predicted')
        axs[0, 1].set_title('Heading Change Comparison')
        axs[0, 1].set_xlabel('Time')
        axs[0, 1].set_ylabel('Heading Change (degrees)')
        axs[0, 1].legend()
        axs[0, 1].grid()
        
        # 속도 오차
        speed_error = np.abs(gt_speed - pred_speed[:len(gt_speed)])
        axs[1, 0].plot(speed_error, 'g-')
        axs[1, 0].set_title('Speed Error')
        axs[1, 0].set_xlabel('Time')
        axs[1, 0].set_ylabel('Speed Error (m/s)')
        axs[1, 0].grid()
        
        # 헤딩 변화량 오차
        heading_error = np.abs(np.degrees(gt_heading_change - pred_heading_change[:len(gt_heading_change)]))
        axs[1, 1].plot(heading_error, 'g-')
        axs[1, 1].set_title('Heading Change Error')
        axs[1, 1].set_xlabel('Time')
        axs[1, 1].set_ylabel('Heading Change Error (degrees)')
        axs[1, 1].grid()
        
        plt.tight_layout()
        plt.show()
        
        # 평균 오차 계산
        mean_speed_error = np.mean(speed_error)
        mean_heading_error = np.mean(heading_error)
        print(f"Mean Speed Error: {mean_speed_error:.4f} m/s")
        print(f"Mean Heading Change Error: {mean_heading_error:.4f} degrees")