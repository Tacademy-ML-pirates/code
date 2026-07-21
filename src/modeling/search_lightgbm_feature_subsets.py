"""
Description:
    상관성이 높은 변수쌍의 제거 조합을 비교하고 Optuna로 LightGBM을 튜닝합니다.

Author:
    김동혁

Source:
    Notion ML database, page 37cbf34c-c2ef-80d4-ace5-dfcfd38c72df
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, roc_auc_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False


print("\n[ 🚀 LightGBM을 활용한 블루리본 예측 최종 모델링 (과적합 방지 Optuna 튜닝) ]")

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

# LightGBM 기준 모델과 동일하게 가중 점수는 유지하고 순수 score 컬럼은 제외
drop_cols = ['카테고리', '매장명', '블루리본 여부', 'idx_family_weighted_score',
             '고급성_중앙값', '쾌적성_중앙값', '감성_중앙값']
# drop_cols = ['카테고리', '매장명', '블루리본 여부', 'idx_family_weighted_score',
#              '고급성_평균', '쾌적성_평균', '감성_평균']
X_base = df.drop(columns=drop_cols)

print(f"✅ 사용된 총 변수 개수: {len(X_base.columns)}개")


# =========================================================================
# 2. Train / Test 분리 및 전처리
# =========================================================================
# random_state, test_size, stratify 고정 (기존 파일과 완벽히 동일한 데이터 분할)
X_train, X_test, y_train, y_test = train_test_split(
    X_base, y, test_size=0.2, random_state=42, stratify=y
)


# =========================================================================
# 3. 상관계수 높은 변수쌍 기준 32개 제거 조합 실험
# =========================================================================

import itertools
from sklearn.model_selection import StratifiedKFold

print("\n[ 🔁 상관계수 높은 변수쌍 기준 32개 조합 실험 시작 ]")

# -------------------------------------------------------------------------
# 3-1. 상관계수 높은 변수쌍 정의
# 각 행에서 왼쪽 변수 또는 오른쪽 변수 중 하나만 제거
# -------------------------------------------------------------------------

corr_pairs = [
    ("idx_anniversary_weighted_score", "idx_luxury_weighted_score"),
    ("idx_date_weighted_score", "idx_ambiance_weighted_score"),
    ("고급성_평균", "감성_평균"),
    # ("고급성_중앙값", "감성_중앙값"),
    ("idx_service_weighted_score", "idx_luxury_weighted_score")
]

print("\n제거 후보 변수쌍:")
for i, (v1, v2) in enumerate(corr_pairs, start=1):
    print(f"{i}. {v1}  vs  {v2}")

# -------------------------------------------------------------------------
# 3-2. CV 설정
# -------------------------------------------------------------------------

N_TRIALS = 50      # 오래 걸리면 20으로 줄이면 됨
CV_SPLITS = 5

cv = StratifiedKFold(
    n_splits=CV_SPLITS,
    shuffle=True,
    random_state=42
)

# -------------------------------------------------------------------------
# 3-3. 전체 조합 생성
# 0이면 변수1 제거, 1이면 변수2 제거
# -------------------------------------------------------------------------

all_combinations = list(itertools.product([0, 1], repeat=len(corr_pairs)))

print(f"\n✅ 총 실험 조합 개수: {len(all_combinations)}개")
print("각 조합마다 상관쌍 5개에서 하나씩 선택하여 제거합니다.")

# 결과 저장용
results = []

# 최적 모델 저장용
best_models = {}


# =========================================================================
# 4. 32개 조합 반복 실행
# =========================================================================

for combo_idx, combo in enumerate(all_combinations, start=1):

    print("\n" + "=" * 90)
    print(f"[진행상황] {combo_idx} / {len(all_combinations)} 번째 조합 실행 중")
    print("=" * 90)

    # ---------------------------------------------------------------------
    # 4-1. 이번 조합에서 제거할 변수 선택
    # ---------------------------------------------------------------------

    selected_drop_cols = []

    for choice, (var1, var2) in zip(combo, corr_pairs):
        if choice == 0:
            selected_drop_cols.append(var1)
        else:
            selected_drop_cols.append(var2)

    # 중복 제거
    # idx_luxury_weighted_score가 두 번 등장하므로 실제 제거 개수는 4개가 될 수도 있음
    selected_drop_cols = list(dict.fromkeys(selected_drop_cols))

    # 실제 X_train에 존재하는 컬럼만 제거
    existing_drop_cols = [col for col in selected_drop_cols if col in X_train.columns]
    missing_drop_cols = [col for col in selected_drop_cols if col not in X_train.columns]

    print("\n이번 조합에서 제거할 변수:")
    for col in existing_drop_cols:
        print(f" - {col}")

    if len(missing_drop_cols) > 0:
        print("\n⚠️ 데이터에 존재하지 않아 제거하지 못한 변수:")
        for col in missing_drop_cols:
            print(f" - {col}")

    # ---------------------------------------------------------------------
    # 4-2. 변수 제거한 데이터 생성
    # ---------------------------------------------------------------------

    X_train_sub = X_train.drop(columns=existing_drop_cols)
    X_test_sub = X_test.drop(columns=existing_drop_cols)

    print(f"\n사용 변수 개수: {X_train_sub.shape[1]}개")
    print(f"실제 제거 변수 개수: {len(existing_drop_cols)}개")

    # ---------------------------------------------------------------------
    # 4-3. Optuna 목적 함수 정의
    # ---------------------------------------------------------------------

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
            X_train_sub,
            y_train,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1
        )

        return np.mean(scores)

    # ---------------------------------------------------------------------
    # 4-4. Optuna 튜닝 실행
    # ---------------------------------------------------------------------

    study = optuna.create_study(
        direction="maximize",
        study_name=f"LGBM_combo_{combo_idx}"
    )

    study.optimize(
        objective_lgbm_regularized,
        n_trials=N_TRIALS,
        show_progress_bar=False
    )

    best_params = study.best_params

    print("\n--- Optuna 튜닝 완료 ---")
    print(f"Best CV ROC-AUC: {study.best_value:.4f}")
    print("Best Params:")
    print(best_params)

    # ---------------------------------------------------------------------
    # 4-5. 최적 파라미터로 최종 모델 학습
    # ---------------------------------------------------------------------

    best_lgbm_model = lgb.LGBMClassifier(
        **best_params,
        objective="binary",
        metric="binary_logloss",
        boosting_type="gbdt",
        random_state=42,
        verbose=-1,
        n_jobs=-1
    )

    best_lgbm_model.fit(X_train_sub, y_train)

    # ---------------------------------------------------------------------
    # 4-6. Train / Test 예측
    # ---------------------------------------------------------------------

    y_train_pred = best_lgbm_model.predict(X_train_sub)
    y_test_pred = best_lgbm_model.predict(X_test_sub)

    y_train_proba = best_lgbm_model.predict_proba(X_train_sub)[:, 1]
    y_test_proba = best_lgbm_model.predict_proba(X_test_sub)[:, 1]

    # ---------------------------------------------------------------------
    # 4-7. 성능 평가
    # ---------------------------------------------------------------------

    train_auc = roc_auc_score(y_train, y_train_proba)
    test_auc = roc_auc_score(y_test, y_test_proba)
    auc_gap = abs(train_auc - test_auc)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    train_f1 = f1_score(y_train, y_train_pred)
    test_f1 = f1_score(y_test, y_test_pred)

    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)

    print("\n[Train vs Test 과적합 검증]")
    print(f"Train ROC-AUC : {train_auc:.4f}")
    print(f"Test ROC-AUC  : {test_auc:.4f}")
    print(f"AUC Gap       : {auc_gap:.4f}")

    print("\n[Test 성능]")
    print(f"Accuracy  : {test_acc:.4f}")
    print(f"Precision : {test_precision:.4f}")
    print(f"Recall    : {test_recall:.4f}")
    print(f"F1-score  : {test_f1:.4f}")

    # ---------------------------------------------------------------------
    # 4-8. 결과 저장
    # ---------------------------------------------------------------------

    result_row = {
        "조합번호": combo_idx,
        "선택패턴": combo,
        "제거변수": ", ".join(existing_drop_cols),
        "제거변수개수": len(existing_drop_cols),
        "사용변수개수": X_train_sub.shape[1],

        "CV_ROC_AUC": study.best_value,

        "Train_ROC_AUC": train_auc,
        "Test_ROC_AUC": test_auc,
        "AUC_Gap": auc_gap,

        "Train_Accuracy": train_acc,
        "Test_Accuracy": test_acc,

        "Train_F1": train_f1,
        "Test_F1": test_f1,

        "Test_Precision": test_precision,
        "Test_Recall": test_recall,

        "Best_Params": best_params
    }

    results.append(result_row)

    best_models[combo_idx] = {
        "model": best_lgbm_model,
        "drop_cols": existing_drop_cols,
        "features": X_train_sub.columns.tolist(),
        "study": study,
        "best_params": best_params
    }


# =========================================================================
# 5. 전체 결과표 생성 및 저장
# =========================================================================

results_df = pd.DataFrame(results)

results_df_sorted = results_df.sort_values(
    by=["Test_ROC_AUC", "AUC_Gap", "Test_F1"],
    ascending=[False, True, False]
).reset_index(drop=True)

print("\n" + "=" * 90)
print("🏆 LightGBM 32개 조합 전체 결과 Top 10")
print("=" * 90)

display_cols = [
    "조합번호",
    "제거변수",
    "제거변수개수",
    "사용변수개수",
    "CV_ROC_AUC",
    "Train_ROC_AUC",
    "Test_ROC_AUC",
    "AUC_Gap",
    "Test_Accuracy",
    "Test_F1",
    "Test_Precision",
    "Test_Recall"
]

print(results_df_sorted[display_cols].head(10))

# CSV 저장
results_df_sorted.to_csv(
    "../data/lgbm_상관변수_32조합_제거실험결과.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n✅ 결과 저장 완료: ../data/lgbm_상관변수_32조합_제거실험결과.csv")
