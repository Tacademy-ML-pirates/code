"""
Description:
    Optuna로 LightGBM 하이퍼파라미터를 탐색하고 분류 성능과 변수 중요도를 평가합니다.

Author:
    김동혁
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    classification_report, roc_auc_score, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False


print("\n[ 🚀 LightGBM + Optuna를 활용한 블루리본 예측 모델링 ]")

# =========================================================================
# 1. 데이터 불러오기 및 X, y 세팅
# =========================================================================
df = pd.read_csv('../data/블루리본_최최종_마참내_v1.2.csv', encoding='cp949')
y = df['블루리본 여부']

# 이미지 모델 결과 컬럼들의 0 값을 NaN으로 변경 (LightGBM 특화 처리)
image_cols = [
    '고급성_평균', '고급성_중앙값',
    '쾌적성_평균', '쾌적성_중앙값',
    '감성_평균', '감성_중앙값'
]
df[image_cols] = df[image_cols].replace(0, np.nan)
print("ℹ️ 이미지 변수 내 0 -> NaN 변환 완료!")

# --- 변수 제거 설정 ---
# 기본적으로 제거할 변수
base_drop_cols = ['카테고리', '매장명', '블루리본 여부', 'idx_family_weighted_score']

final_drop_cols = list(set(base_drop_cols))
X_base = df.drop(columns=final_drop_cols)

print("\n✅ 다음 변수들을 제거하고 모델링을 시작합니다:")
for col in sorted(final_drop_cols):
    if col in df.columns:
        print(f" - {col}")
print(f"✅ 사용된 총 변수 개수: {len(X_base.columns)}개")


# =========================================================================
# 2. Train / Test 분리
# =========================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_base, y, test_size=0.2, random_state=42, stratify=y
)


# =========================================================================
# 3. Optuna를 사용한 하이퍼파라미터 튜닝
# =========================================================================
print("\n[ 🔎 Optuna를 사용하여 최적의 하이퍼파라미터 탐색 시작 ]")

N_TRIALS = 100      # 탐색 횟수 (시간이 오래 걸리면 50으로 줄여서 테스트)
CV_SPLITS = 5

cv = StratifiedKFold(
    n_splits=CV_SPLITS,
    shuffle=True,
    random_state=42
)

def objective_lgbm_regularized(trial):
    params = {
        # 기본 설정
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "random_state": 42,
        "verbose": -1,
        "n_jobs": -1,

        # 모델 복잡도 제어
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 7, 31),
        "max_depth": trial.suggest_int("max_depth", 2, 6),

        # 과적합 방지용 샘플링
        "subsample": trial.suggest_float("subsample", 0.5, 0.85),
        "subsample_freq": trial.suggest_int("subsample_freq", 1, 5),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.85),

        # 리프 최소 데이터 수
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.001, 10.0, log=True),

        # 규제
        "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 10.0, log=True),

        # 분할 이득 제한
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0)
    }

    model = lgb.LGBMClassifier(**params)

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1
    )

    return np.mean(scores)

study = optuna.create_study(
    direction="maximize",
    study_name="LGBM_Optimization"
)

study.optimize(
    objective_lgbm_regularized,
    n_trials=N_TRIALS,
    show_progress_bar=True
)

best_params = study.best_params

print("\n" + "=" * 90)
print("🏆 Optuna 튜닝 완료")
print(f"Best CV ROC-AUC: {study.best_value:.4f}")
print("\nBest Hyperparameters:")
print(best_params)
print("=" * 90)

# =========================================================================
# 4. 최적 파라미터로 최종 모델 학습 및 평가
# =========================================================================

final_model = lgb.LGBMClassifier(
    **best_params,
    objective="binary",
    metric="binary_logloss",
    boosting_type="gbdt",
    random_state=42,
    verbose=-1,
    n_jobs=-1
)

final_model.fit(X_train, y_train)

y_train_pred = final_model.predict(X_train)
y_test_pred = final_model.predict(X_test)

y_train_proba = final_model.predict_proba(X_train)[:, 1]
y_test_proba = final_model.predict_proba(X_test)[:, 1]

train_auc = roc_auc_score(y_train, y_train_proba)
test_auc = roc_auc_score(y_test, y_test_proba)
auc_gap = abs(train_auc - test_auc)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print("\n[ 🎯 Train vs Test 과적합 검증 ]")
print(f"Train ROC-AUC : {train_auc:.4f}")
print(f"Test ROC-AUC  : {test_auc:.4f}")
print(f"AUC Gap       : {auc_gap:.4f} (작을수록 안정적)")
print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy : {test_acc:.4f}")

print("\n[ 📊 최종 Test 실전 성능 ]")
print(f"Accuracy  : {test_acc:.4f}")
print(classification_report(y_test, y_test_pred, zero_division=0))

# =========================================================================
# 5. 특성 중요도 시각화
# =========================================================================

fig, ax = plt.subplots(figsize=(12, 8))
lgb.plot_importance(final_model, ax=ax, max_num_features=20, height=0.8,
                    importance_type='gain', title='LightGBM Feature Importance (Gain)')
plt.tight_layout()
plt.show()
