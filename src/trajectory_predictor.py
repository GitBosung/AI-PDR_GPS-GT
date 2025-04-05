import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class TrajectoryPredictor:
    """
    학습된 모델을 사용하여 새로운 데이터의 이동 경로를 예측하고 시각화하는 클래스.
    
    주요 기능:
      - 센서 데이터 추출 및 스케일링 (모델 학습 시 사용한 scaler 적용)
      - 슬라이딩 윈도우 방식으로 입력 데이터를 생성하여 모델 예측 수행
      - 예측 결과를 바탕으로 경로 계산 및 다양한 시각화 제공
    """
    def __init__(self, model, scaler_acc, scaler_gyro, scaler_ori, window_size=50):
        self.model = model
        self.scaler_acc = scaler_acc
        self.scaler_gyro = scaler_gyro
        self.scaler_ori = scaler_ori
        self.window_size = window_size

    def _prepare_sensor_data(self, df, sensor_columns):
        """
        데이터프레임에서 센서 데이터를 추출하고, 각 센서별로 스케일링합니다.
        """
        sensor_data = df[sensor_columns].copy().astype(np.float32)
        
        # Accelerometer 스케일링
        acc_data = sensor_data.iloc[:, 0:3].values
        acc_scaled = self.scaler_acc.transform(acc_data)
        sensor_data.iloc[:, 0:3] = acc_scaled

        # Gyroscope 스케일링
        gyro_data = sensor_data.iloc[:, 3:6].values
        gyro_scaled = self.scaler_gyro.transform(gyro_data)
        sensor_data.iloc[:, 3:6] = gyro_scaled

        # Orientation 스케일링
        ori_data = sensor_data.iloc[:, 6:9].values
        ori_scaled = self.scaler_ori.transform(ori_data)
        sensor_data.iloc[:, 6:9] = ori_scaled

        return sensor_data

    def predict_and_plot_trajectory(self, df):
        """
        새로운 데이터에 대해 모델 예측을 수행하고, 이동 경로 및 속도/헤딩 변화량 분포를 시각화합니다.
        """
        sensor_columns = ['Accelerometer x', 'Accelerometer y', 'Accelerometer z',
                          'Gyroscope x', 'Gyroscope y', 'Gyroscope z',
                          'Orientation x', 'Orientation y', 'Orientation z']
        
        # 센서 데이터 추출 및 스케일링
        sensor_data = self._prepare_sensor_data(df, sensor_columns)
        
        # 슬라이딩 윈도우 방식으로 입력 데이터 생성
        X_test_new = []
        for i in range(0, len(sensor_data) - self.window_size, self.window_size):
            window = sensor_data.iloc[i:i + self.window_size].values
            X_test_new.append(window)
        X_test_new = np.array(X_test_new).astype(np.float32)
        
        # 모델 예측 수행: 출력은 [속도, 헤딩 변화량]
        Y_pred = self.model.predict(X_test_new)
        
        # 예측 결과를 바탕으로 이동 경로 계산
        x, y = 0, 0
        trajectory_x, trajectory_y = [x], [y]
        U, V = [] , []
        heading = 0
        for speed, heading_change in Y_pred:
            heading += heading_change
            dx = speed * np.cos(heading)
            dy = speed * np.sin(heading)
            U.append(dx)
            V.append(dy)
            x += dx
            y += dy
            trajectory_x.append(x)
            trajectory_y.append(y)
        
        # 이동 경로 시각화 (화살표 표시)
        plt.figure(figsize=(8, 6))
        plt.quiver(trajectory_x[:-1], trajectory_y[:-1], U, V, angles='xy', scale_units='xy', scale=1, alpha=0.7)
        plt.plot(trajectory_x[0], trajectory_y[0], 'ro', markersize=8, label="Start")
        plt.plot(trajectory_x[-1], trajectory_y[-1], 'go', markersize=8, label="End")
        plt.xlabel('Easting (m)')
        plt.ylabel('Northing (m)')
        plt.title('Predicted Movement Trajectory')
        plt.legend()
        plt.grid()
        plt.axis('equal')
        plt.show()
        
        # 추가: 속도와 헤딩 변화량 분포 시각화
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
