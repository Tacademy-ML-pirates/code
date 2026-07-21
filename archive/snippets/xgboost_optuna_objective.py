"""
Description:
    XGBoost 정규화 파라미터 범위를 정의한 미완성 Optuna 목적함수 실험 조각입니다.

Author:
    이용현

Source:
    Notion ML database, page 37cbf34c-c2ef-801b-9493-e8c3d04ab361
"""

def objective(trial):
    params = {
        # 1. 나무 깊이는 3~5로 제한 (6 이상 올라가지 못하게 물리적 차단)
        'max_depth': trial.suggest_int('max_depth', 3, 5),

        # 2. 자식 노드의 최소 샘플 수를 5개 이상으로 엄격하게 제한
        'min_child_weight': trial.suggest_int('min_child_weight', 5, 12),

        # 3. 샘플링 비율을 60%~80%로 제한하여 무작위성 확보
        'subsample': trial.suggest_float('subsample', 0.6, 0.8),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.8),

        # 4. 가지치기 통행세(Gamma)를 높여 잔가지 차단
        'gamma': trial.suggest_float('gamma', 1.0, 5.0),

        # 5. L1, L2 규제 값을 주어 가중치가 튀는 것을 방지
        'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 5.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 10.0),

        # 기존 세팅 유지
        'n_estimators': 150,
        'learning_rate': 0.05,
        'random_state': 42,
        'eval_metric': 'logloss'
    }

    # 이 하단에 기존 cross_val_score 및 리턴 코드는 그대로 유지하시면 됩니다.
