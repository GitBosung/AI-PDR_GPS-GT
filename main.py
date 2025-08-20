# main.py
from src.data_processor import DataProcessor
from src.model_trainer import ModelTrainer
from src.trajectory_predictor import TrajectoryPredictor

import numpy as np
import os
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path

# 현재 main.py 파일의 디렉토리를 기준으로 프로젝트 루트 경로 설정
BASE_DIR = os.getcwd()

# 시드 값 고정
SEED = 20205136

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

config = {
    'looking_left05.csv': {'skiprows': 500, 'flag': True,  'zone': 52},
    'looking_lr02.csv':   {'skiprows': 500, 'flag': True,  'zone': 52},
    'UTAH_looking_01.csv':{'skiprows': 1000,'flag': True,  'zone': 12},
    'UTAH_looking_02.csv':{'skiprows': 1000,'flag': True,  'zone': 12},
    'UTAH_looking_03.csv':{'skiprows': 1000,'flag': True,  'zone': 12},
    'UTAH_looking_04.csv':{'skiprows': 1000,'flag': True,  'zone': 12},
}
default_config = {'skiprows': 500, 'flag': False, 'zone': 52}

def main():
    
    window_size = 200
    # ============================================================
    # 1. 학습 데이터 로딩 및 전처리
    # ============================================================
    learn_data_paths = [
    os.path.join(BASE_DIR, 'data', 'learn_data', 'looking','looking_left01.csv'),
    #os.path.join(BASE_DIR, 'data', 'learn_data', 'looking','looking_left02.csv'),
    os.path.join(BASE_DIR, 'data', 'learn_data', 'looking','looking_left03.csv'),
    os.path.join(BASE_DIR, 'data', 'learn_data', 'looking','looking_left04.csv'),
    os.path.join(BASE_DIR, 'data', 'learn_data', 'looking','looking_left05.csv'), 

    os.path.join(BASE_DIR, 'data', 'learn_data', 'looking','looking_right01.csv'),
    os.path.join(BASE_DIR, 'data', 'learn_data', 'looking','looking_right02.csv'),
    os.path.join(BASE_DIR, 'data', 'learn_data', 'looking','looking_right03.csv'),

    os.path.join(BASE_DIR, 'data', 'learn_data', 'looking','looking_lr01.csv'),
    os.path.join(BASE_DIR, 'data', 'learn_data', 'looking','looking_lr02.csv'),
    

    os.path.join(BASE_DIR, 'data', 'learn_data', 'swing', 'swing_left01.csv'),
    os.path.join(BASE_DIR, 'data', 'learn_data', 'swing', 'swing_left02.csv'),
    os.path.join(BASE_DIR, 'data', 'learn_data', 'swing', 'swing_left03.csv'),
    
    os.path.join(BASE_DIR, 'data', 'learn_data', 'swing', 'swing_right01.csv'),
    os.path.join(BASE_DIR, 'data', 'learn_data', 'swing', 'swing_right02.csv'),
    os.path.join(BASE_DIR, 'data', 'learn_data', 'swing', 'swing_right03.csv'),

    os.path.join(BASE_DIR, 'data', 'learn_data', 'swing', 'swing_lr01.csv'),
    
    #--- 유타에서 수집한 데이터 ---#
    
    # os.path.join(BASE_DIR, 'data', 'learn_data', 'UTAH_Looking_1.csv'),
    # os.path.join(BASE_DIR, 'data', 'learn_data', 'UTAH_test01.csv'),
    # os.path.join(BASE_DIR, 'data', 'learn_data', 'UTAH_looking_01.csv'),
    # os.path.join(BASE_DIR, 'data', 'learn_data', 'UTAH_looking_02.csv'),
    # os.path.join(BASE_DIR, 'data', 'learn_data', 'UTAH_looking_03.csv'),
    # os.path.join(BASE_DIR, 'data', 'learn_data', 'UTAH_looking_04.csv'),
    ]
    
    test_paths = [
        
        os.path.join(BASE_DIR, 'data', 'test_data', 'looking', 'test_looking_l01.csv'),
        os.path.join(BASE_DIR, 'data', 'test_data', 'looking', 'test_looking_l02.csv'),
        os.path.join(BASE_DIR, 'data', 'test_data', 'looking', 'test_looking_l03.csv'),
        os.path.join(BASE_DIR, 'data', 'test_data', 'looking', 'test_looking_l04.csv'),
        os.path.join(BASE_DIR, 'data', 'test_data', 'looking', 'test_looking_l05.csv'),
        os.path.join(BASE_DIR, 'data', 'test_data', 'looking', 'test_looking_l06.csv'),
        os.path.join(BASE_DIR, 'data', 'test_data', 'looking', 'test_looking_l07.csv'),
        os.path.join(BASE_DIR, 'data', 'test_data', 'looking', 'test_looking_l08.csv'),
        os.path.join(BASE_DIR, 'data', 'test_data', 'looking', 'test_looking_l09.csv'),
        
        os.path.join(BASE_DIR, 'data', 'test_data', 'looking', 'test_looking_r01.csv'),
        os.path.join(BASE_DIR, 'data', 'test_data', 'looking', 'test_looking_r02.csv'),
        os.path.join(BASE_DIR, 'data', 'test_data', 'looking', 'test_looking_r03.csv'),
        os.path.join(BASE_DIR, 'data', 'test_data', 'looking', 'test_looking_r04.csv'),
        os.path.join(BASE_DIR, 'data', 'test_data', 'looking', 'test_looking_r05.csv'),
        os.path.join(BASE_DIR, 'data', 'test_data', 'looking', 'test_looking_r06.csv'),
        os.path.join(BASE_DIR, 'data', 'test_data', 'looking', 'test_looking_r07.csv'),
        os.path.join(BASE_DIR, 'data', 'test_data', 'looking', 'test_looking_r08.csv'),
        os.path.join(BASE_DIR, 'data', 'test_data', 'looking', 'test_looking_r09.csv'),
        os.path.join(BASE_DIR, 'data', 'test_data', 'looking', 'test_looking_r10.csv'),
        
        os.path.join(BASE_DIR, 'data', 'test_data', 'looking', 'test_looking_3loop_l01.csv'),
        os.path.join(BASE_DIR, 'data', 'test_data', 'looking', 'test_looking_3loop_r01.csv'),
        
        os.path.join(BASE_DIR, 'data', 'test_data', 'looking', 'test_looking_5loop_l01.csv'),
        os.path.join(BASE_DIR, 'data', 'test_data', 'looking', 'test_looking_5loop_l02.csv'),
        
        os.path.join(BASE_DIR, 'data', 'test_data', 'swing', 'test_swing_l01.csv'),
        os.path.join(BASE_DIR, 'data', 'test_data', 'swing', 'test_swing_l02.csv'),
        os.path.join(BASE_DIR, 'data', 'test_data', 'swing', 'test_swing_l03.csv'),
        os.path.join(BASE_DIR, 'data', 'test_data', 'swing', 'test_swing_l04.csv'),
        os.path.join(BASE_DIR, 'data', 'test_data', 'swing', 'test_swing_l05.csv'),
        
        os.path.join(BASE_DIR, 'data', 'test_data', 'swing', 'test_swing_r01.csv'),
        os.path.join(BASE_DIR, 'data', 'test_data', 'swing', 'test_swing_r02.csv'),
    ]

    
    X_list, Y_list = [], []
    df_list = []

    for path in learn_data_paths:
        if not os.path.exists(path):
            print(f"학습 파일을 찾을 수 없습니다: {path}")
            continue

        # 파일명으로 옵션 선택
        fname = Path(path).name
        opts = config.get(fname, default_config)

        # 언팩하여 함수 호출
        df_all, x, y = DataProcessor.load_and_preprocess_csv(path, **opts)

        df_list.append(df_all)
        X_list.append(x)
        Y_list.append(y)

    if not X_list:
        print("유효한 윈도우가 하나도 생성되지 않았습니다.")
        return

    # 4) numpy 배열로 변환
    X = np.concatenate(X_list, axis=0)  # (batch, window_size, feature_dim)
    Y = np.concatenate(Y_list, axis=0)  # (batch, 2)

    # ============================================================
    # 이상치 제거: 헤딩 변화량이 100도 초과하는 샘플 제외
    # ============================================================
    # degrees_psi = np.degrees(Y[:, 1])
    # mask = np.abs(degrees_psi) <= 100
    # removed = np.sum(~mask)
    # print(f"제거된 샘플 개수: {removed}")

    # X = X[mask]
    # Y = Y[mask]

    # # ============================================================
    # # 5. 데이터 분포 시각화 (optional)
    # # ============================================================
    delta_l = Y[:, 0]     # Distance change
    delta_psi = Y[:, 1]   # Heading change

    plt.plot(np.degrees(delta_psi))
    plt.title('Heading Change (degrees)')
    plt.xlabel('Sample Index')
    plt.ylabel('Heading Change (degrees)')
    plt.grid(True)
    plt.show()

    def plot_pdf(data, label_name):
        plt.figure()
        plt.hist(data, bins=50, density=True, alpha=0.7)
        plt.xlabel(f'{label_name}')
        plt.ylabel('Probability Density')
        plt.title(f'PDF of {label_name}')
        plt.grid(True)
        plt.show()

    plot_pdf(delta_l, 'Speed (m/s)')
    plot_pdf(np.degrees(delta_psi), 'Δψ (degrees)')

    # ============================================================
    # 6. 모델 학습
    # ============================================================
    total_samples, window_size, num_features = X.shape
    trainer = ModelTrainer(window_size, num_features, epochs=15, batch_size=256)
    history = trainer.train_model(X, Y)
    trainer.plot_training_history(history)

    model_path = trainer.save_model()
    print("모델이 저장되었습니다:", model_path)

    # ============================================================
    # 7. 학습 데이터 GT vs 예측 비교 (TrajectoryPredictor 사용)
    # ============================================================
    predictor = TrajectoryPredictor(
        trainer.model,
        trainer.sensor_scalers,
        trainer.y_speed_scaler,
        trainer.y_hc_scaler,
        window_size
    )

    # df_list와 Y_list를 zip으로 묶어서, 각각 GT(Y)와 함께 전달
    for df_learn, Y_learn in zip(df_list, Y_list):
        predictor.compare_trajectories(df_learn)

    # ============================================================
    # 8. 테스트 데이터에 대해 예측 및 이동 경로 시각화
    # ============================================================


    for test_path in test_paths:
        if os.path.exists(test_path):
            df_test = DataProcessor.load_and_preprocess_csv_test(test_path, skiprows=100)
            predictor.predict_and_plot_trajectory(df_test)
            predictor.predict_and_plot_trajectory(df_test, True)
        else:
            print(f"테스트 파일을 찾을 수 없습니다: {test_path}")


if __name__ == '__main__':
    main()
