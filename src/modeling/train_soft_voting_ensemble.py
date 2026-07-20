"""
Description:
    LightGBM과 XGBoost 예측 확률을 결합하는 소프트 보팅 앙상블을 학습·평가합니다.

Author:
    김동혁
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import optuna
import itertools

from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    classification_report, roc_auc_score, f1_score, precision_recall_curve
)

import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

print("\n[ 🚀 LightGBM + XGBoost 앙상블 (소프트 보팅) 최종 모델링 ]")

# =========================================================================
# 1. 데이터 불러오기 및 X, y 세팅
# =========================================================================
df = pd.read_csv('../data/블루리본_최최종_마참내_v1.2.csv', encoding='cp949')
y = df['블루리본 여부']

image_cols = [
    '고급성_평균', '고급성_중앙값','쾌적성_중앙값','감성_중앙값',
    '쾌적성_평균', '감성_평균'
]
df[image_cols] = df[image_cols].replace(0, np.nan)
print("ℹ️ 이미지 변수 내 0 -> NaN 변환 완료!")

drop_cols = ['카테고리', '매장명', '블루리본 여부','고급성_중앙값','쾌적성_중앙값','감성_중앙값',
            'idx_family_weighted_score']
X_base = df.drop(columns=drop_cols)

print(f"✅ 사용된 총 변수 개수: {len(X_base.columns)}개")

# =========================================================================
# 2. Train / Test 분리
# =========================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_base, y, test_size=0.2, random_state=42, stratify=y
)

# =========================================================================
# 3. 조합 및 CV 설정
# =========================================================================
corr_pairs = [
    ("idx_anniversary_weighted_score", "idx_luxury_weighted_score"),
    ("idx_date_weighted_score", "idx_ambiance_weighted_score"),
    ("고급성_평균", "감성_평균"),
    ("idx_service_weighted_score", "idx_luxury_weighted_score")
]

N_TRIALS = 20  # 두 모델을 돌리므로 시간을 위해 20으로 단축 (필요시 50으로 수정)
CV_SPLITS = 5

cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=42)
all_combinations = list(itertools.product([0, 1], repeat=len(corr_pairs)))

print(f"\n✅ 총 실험 조합 개수: {len(all_combinations)}개 (LGBM, XGB 각각 최적화 진행)")

best_lgbm_score = 0
best_lgbm_params = None
best_lgbm_drop_cols = None

best_xgb_score = 0
best_xgb_params = None
best_xgb_drop_cols = None

# =========================================================================
# 4. 각 모델별 최적의 변수 조합 및 하이퍼파라미터 찾기
# =========================================================================
optuna.logging.set_verbosity(optuna.logging.WARNING) # Optuna 로그 숨기기 (진행상황만 출력)

for combo_idx, combo in enumerate(all_combinations, start=1):
    print(f"\n[ 🔄 조합 {combo_idx}/{len(all_combinations)} 탐색 중... ]")

    selected_drop_cols = list(dict.fromkeys([var1 if choice == 0 else var2 for choice, (var1, var2) in zip(combo, corr_pairs)]))
    existing_drop_cols = [col for col in selected_drop_cols if col in X_train.columns]

    X_train_sub = X_train.drop(columns=existing_drop_cols)

    # --- LightGBM 목적 함수 ---
    def objective_lgbm(trial):
        params = {
            "objective": "binary", "metric": "binary_logloss", "random_state": 42, "verbose": -1, "n_jobs": -1,
            "n_estimators": trial.suggest_int("n_estimators", 100, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 10, 31),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            "subsample": trial.suggest_float("subsample", 0.5, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9)
        }
        model = lgb.LGBMClassifier(**params)
        # 윈도우 프리징 방지를 위해 n_jobs=1 유지
        return np.mean(cross_val_score(model, X_train_sub, y_train, cv=cv, scoring="roc_auc", n_jobs=1))

    # --- XGBoost 목적 함수 ---
    def objective_xgb(trial):
        params = {
            "objective": "binary:logistic", "eval_metric": "logloss", "random_state": 42, "n_jobs": -1, "tree_method": "hist",
            "n_estimators": trial.suggest_int("n_estimators", 100, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "min_child_weight": trial.suggest_int("min_child_weight", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9)
        }
        model = xgb.XGBClassifier(**params)
        # 윈도우 프리징 방지를 위해 n_jobs=1 유지
        return np.mean(cross_val_score(model, X_train_sub, y_train, cv=cv, scoring="roc_auc", n_jobs=1))

    # 튜닝 실행
    study_lgbm = optuna.create_study(direction="maximize")
    study_lgbm.optimize(objective_lgbm, n_trials=N_TRIALS)

    study_xgb = optuna.create_study(direction="maximize")
    study_xgb.optimize(objective_xgb, n_trials=N_TRIALS)

    # 최고 점수 갱신 시 기록
    if study_lgbm.best_value > best_lgbm_score:
        best_lgbm_score = study_lgbm.best_value
        best_lgbm_params = study_lgbm.best_params
        best_lgbm_drop_cols = existing_drop_cols

    if study_xgb.best_value > best_xgb_score:
        best_xgb_score = study_xgb.best_value
        best_xgb_params = study_xgb.best_params
        best_xgb_drop_cols = existing_drop_cols


# =========================================================================
# 5. 최종 앙상블 모델 학습 및 예측
# =========================================================================
print("\n" + "=" * 90)
print(f"🥇 최고 성능 LightGBM CV AUC: {best_lgbm_score:.4f}")
print(f"🥇 최고 성능 XGBoost CV AUC : {best_xgb_score:.4f}")
print("=" * 90)

# LGBM 최종 학습
X_train_lgbm = X_train.drop(columns=best_lgbm_drop_cols)
X_test_lgbm = X_test.drop(columns=best_lgbm_drop_cols)

final_lgbm = lgb.LGBMClassifier(**best_lgbm_params, objective="binary", random_state=42, verbose=-1, n_jobs=-1)
final_lgbm.fit(X_train_lgbm, y_train)

# XGB 최종 학습
X_train_xgb = X_train.drop(columns=best_xgb_drop_cols)
X_test_xgb = X_test.drop(columns=best_xgb_drop_cols)

final_xgb = xgb.XGBClassifier(**best_xgb_params, objective="binary:logistic", random_state=42, n_jobs=-1, tree_method="hist")
final_xgb.fit(X_train_xgb, y_train)

# 각 모델의 예측 확률 계산 (Train / Test)
train_proba_lgbm = final_lgbm.predict_proba(X_train_lgbm)[:, 1]
train_proba_xgb = final_xgb.predict_proba(X_train_xgb)[:, 1]
test_proba_lgbm = final_lgbm.predict_proba(X_test_lgbm)[:, 1]
test_proba_xgb = final_xgb.predict_proba(X_test_xgb)[:, 1]

# =========================================================================
# 6. 최적 가중치(Weight) 및 임계값(Threshold) 찾기
# =========================================================================
print("\n[ 🔍 최적의 앙상블 가중치 비율 및 임계값 탐색 중... ]")

best_weight_lgbm = 0.5
best_threshold = 0.5
best_train_f1 = 0

# LightGBM 가중치를 0.0부터 1.0까지 0.05 단위로 테스트 (총 21개 비율 테스트)
weights = np.linspace(0.0, 1.0, 21)

for w_lgbm in weights:
    w_xgb = 1.0 - w_lgbm

    # 현재 비율로 예측 확률 계산
    temp_train_proba = (train_proba_lgbm * w_lgbm) + (train_proba_xgb * w_xgb)

    # 현재 섞인 확률 분포에서 최적의 Threshold 찾기 (Train 기준)
    precisions, recalls, thresholds = precision_recall_curve(y_train, temp_train_proba)
    # thresholds 길이와 맞추기 위해 마지막 값 제외
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)

    if len(f1_scores) > 0:
        max_f1_idx = np.argmax(f1_scores)
        temp_best_f1 = f1_scores[max_f1_idx]
        temp_best_thresh = thresholds[max_f1_idx]

        # 최고 Train F1 갱신 시 기록
        if temp_best_f1 > best_train_f1:
            best_train_f1 = temp_best_f1
            best_weight_lgbm = w_lgbm
            best_threshold = temp_best_thresh

best_weight_xgb = 1.0 - best_weight_lgbm

print(f"🥇 최적 앙상블 비율 - LightGBM: {best_weight_lgbm*100:.0f}%, XGBoost: {best_weight_xgb*100:.0f}%")

print(f"\n💡 앙상블 모델 최적 예측 기준점(Threshold): {best_threshold:.4f} (기본 0.5에서 조정됨)")

# 🤝 찾아낸 최적 가중치로 최종 예측 확률 고정
train_proba_ensemble = (train_proba_lgbm * best_weight_lgbm) + (train_proba_xgb * best_weight_xgb)
test_proba_ensemble = (test_proba_lgbm * best_weight_lgbm) + (test_proba_xgb * best_weight_xgb)

# 찾은 Threshold를 Test 데이터에 적용하여 0 또는 1로 분류
y_train_pred_ens = (train_proba_ensemble >= best_threshold).astype(int)
y_test_pred_ens = (test_proba_ensemble >= best_threshold).astype(int)

# 평가지표 산출
train_auc_ens = roc_auc_score(y_train, train_proba_ensemble)
test_auc_ens = roc_auc_score(y_test, test_proba_ensemble)
auc_gap = abs(train_auc_ens - test_auc_ens)

print("\n[ 🎯 앙상블 모델 Train vs Test 과적합 검증 ]")
print(f"Train ROC-AUC : {train_auc_ens:.4f}")
print(f"Test ROC-AUC  : {test_auc_ens:.4f}")
print(f"AUC Gap       : {auc_gap:.4f} (작을수록 안정적)")

print("\n[ 📊 앙상블 모델 최종 Test 실전 성능 ]")
print(f"Accuracy  : {accuracy_score(y_test, y_test_pred_ens):.4f}")
print(f"Precision : {precision_score(y_test, y_test_pred_ens):.4f}")
print(f"Recall    : {recall_score(y_test, y_test_pred_ens):.4f}")
print(f"F1-score  : {f1_score(y_test, y_test_pred_ens):.4f}")

print("\n[ 상세 분류 리포트 ]")
print(classification_report(y_test, y_test_pred_ens, zero_division=0))

# =========================================================================
# 7. 예측 확률 시각화 (선택사항)
# =========================================================================
plt.figure(figsize=(10, 6))
sns.kdeplot(test_proba_lgbm, label='LightGBM 확률', fill=True, color='skyblue', alpha=0.3)
sns.kdeplot(test_proba_xgb, label='XGBoost 확률', fill=True, color='lightgreen', alpha=0.3)
sns.kdeplot(test_proba_ensemble, label=f'앙상블({best_weight_lgbm*100:.0f}:{best_weight_xgb*100:.0f}) 확률', fill=True, color='coral', alpha=0.6)

plt.axvline(best_threshold, color='red', linestyle='--', label=f'최적 Threshold ({best_threshold:.2f})')
plt.title("모델별 예측 확률 분포 비교 및 앙상블 효과")
plt.xlabel("블루리본 선정 예측 확률")
plt.ylabel("밀도")
plt.legend()
plt.tight_layout()
plt.show()
