from src.data_processor import DataProcessor
from src.model_trainer import ModelTrainer
from src.trajectory_predictor import TrajectoryPredictor
import numpy as np
import os
import tensorflow as tf
import random

# 시드 값 고정
SEED = 11217  # 011217에서 앞의 0을 제외한 값
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 현재 main.py 파일의 디렉토리를 기준으로 프로젝트 루트 경로 설정
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#ipynb 환경인 경우 해당 코드를 사용함
BASE_DIR = os.getcwd()

def main():
    # ============================================================
    # 1. 학습 데이터 로딩 및 전처리
    # ============================================================
    learn_data_paths = [
        os.path.join(BASE_DIR, 'data', 'learn_data', 'Basket_1.csv'),
        os.path.join(BASE_DIR, 'data', 'learn_data', 'Basket_2.csv'),
        os.path.join(BASE_DIR, 'data', 'learn_data', 'Soccer_looking1.csv'),
        os.path.join(BASE_DIR, 'data', 'learn_data', 'Soccer_looking3.csv'),
        os.path.join(BASE_DIR, 'data', 'learn_data', 'Soccer_looking4.csv'),
        os.path.join(BASE_DIR, 'data', 'learn_data', 'Soccer_looking5m_01.csv'),
        os.path.join(BASE_DIR, 'data', 'learn_data', 'Soccer_looking5m_02.csv'),
        os.path.join(BASE_DIR, 'data', 'learn_data', 'Soccer_looking5m_03.csv'),
        os.path.join(BASE_DIR, 'data', 'learn_data', 'Soccer_swing5m_01.csv'),
        os.path.join(BASE_DIR, 'data', 'learn_data', 'Soccer_swing5m_02.csv'),
        os.path.join(BASE_DIR, 'data', 'learn_data', 'Soccer_swing5m_03.csv'),
        os.path.join(BASE_DIR, 'data', 'learn_data', 'Soccer_swing1.csv'),
        os.path.join(BASE_DIR, 'data', 'learn_data', 'Soccer_swing2.csv'),
        os.path.join(BASE_DIR, 'data', 'learn_data', 'Soccer_swing3.csv'),
        os.path.join(BASE_DIR, 'data', 'learn_data', 'Soccer_swing4.csv'),
    ]

    df_list = []
    for path in learn_data_paths:
        if os.path.exists(path):
            df = DataProcessor.load_and_preprocess_csv(path)
            df_list.append(df)
        else:
            print(f"파일을 찾을 수 없습니다: {path}")
    
    # ============================================================
    # 2. 슬라이딩 윈도우 방식으로 X, Y 데이터 생성
    # ============================================================
    X_list, Y_list = [], []
    for df in df_list:
        X_sub, Y_sub = DataProcessor.make_XY_using_dataframe(df)
        X_list.append(X_sub)
        Y_list.append(Y_sub)
    
    # 만약 데이터를 불러오지 못했다면 종료
    if not X_list:
        print("학습 데이터를 불러오지 못했습니다.")
        return

    X = np.concatenate(X_list, axis=0)  # 최종 X: (전체 샘플 수, window_size, 센서 채널 수)
    Y = np.concatenate(Y_list, axis=0)  # 최종 Y: (전체 샘플 수, 2) → [속도, 헤딩 변화량]
    
    print('Data Size: ', X.shape, Y.shape)
    
    # 정답 레이블 분포 시각화
    DataProcessor.plot_label_distribution(Y, save_path=os.path.join(BASE_DIR, 'plots', 'label_distribution.png'))
    
    # ============================================================
    # 3. 모델 학습
    # ============================================================
    total_samples, window_size, num_features = X.shape
    trainer = ModelTrainer(window_size, num_features, epochs=100, batch_size=128)
    history, _ = trainer.train_model(X, Y)
    trainer.plot_training_history(history)
    
    # 모델 저장
    model_path = trainer.save_model()
    print("모델이 저장되었습니다:", model_path)
    
    # ============================================================
    # 4. 테스트 데이터에 대해 예측 및 이동 경로 시각화
    # ============================================================
    for df_learn in df_list:
            predictor = TrajectoryPredictor(trainer.model, trainer.scaler)
            # 예측 경로 시각화
            predictor.compare_trajectories(df_learn)
    
    test_paths = [
        os.path.join(BASE_DIR, 'data', 'test_data', '3f_1.csv'),
        os.path.join(BASE_DIR, 'data', 'test_data', '3f_2.csv'),
        os.path.join(BASE_DIR, 'data', 'test_data', '3f_swing1.csv'),
        os.path.join(BASE_DIR, 'data', 'test_data', '3f_swing2.csv'),
        os.path.join(BASE_DIR, 'data', 'test_data', '3f_looking01.csv'),
        os.path.join(BASE_DIR, 'data', 'test_data', '3f_looking02.csv')
    ]
    for test_path in test_paths:
        if os.path.exists(test_path):
            df_test = DataProcessor.load_and_preprocess_csv(test_path, skiprows=50)
            predictor = TrajectoryPredictor(trainer.model, trainer.scaler)
            
            # 예측 경로 시각화
            predictor.predict_and_plot_trajectory(df_test)
            

        else:
            print(f"테스트 파일을 찾을 수 없습니다: {test_path}")

if __name__ == '__main__':
    main()

