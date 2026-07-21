"""
Description:
    리뷰 감성 특징을 결합하고 LightGBM 하이퍼파라미터를 탐색했던 과거 실험 코드입니다.

Author:
    김동혁
"""

import pandas as pd
import numpy as np

all_df = pd.read_csv('../data/리뷰_감정분석.csv')
all_df.head()
all_df.columns
global_mean = all_df['positive_score'].mean()


df_koelectra = pd.read_csv('../data/KoELECTRA_X_feature_v2.2_ratio.csv', encoding='utf-8-sig')

keywords = ["family", "anniversary", "date", "distance", "revisit",
             "service", "ambiance", "view", "taste",
             "price", "waiting", "luxury", "disappoint"]

score_cols = [c for c in df_koelectra.columns if c.endswith("_score")]
count_cols = [c for c in df_koelectra.columns if c.endswith("_count")]
m = 20

bayes_cols_to_keep = ['매장명'] # 병합을 위해 키(Key) 변수인 매장명 보관
for score_col in score_cols:
    base = score_col[:-6]  # "_score" 제거
    count_col = base + "_count"
    bayes_col = base + "_bayes"

    if count_col in df_koelectra.columns:
        df_koelectra[bayes_col] = (
            (df_koelectra[count_col] * df_koelectra[score_col] + m * global_mean)
            / (df_koelectra[count_col] + m)
        )
        bayes_cols_to_keep.append(bayes_col)

print(f"✅ 계산된 Bayes 변수 개수: {len(bayes_cols_to_keep) - 1}개")




#######
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
# import optuna.visualization as vis

# 한글 폰트 설정
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False


print("\n[ 🚀 LightGBM + Optuna를 활용한 블루리본 예측 모델링 ]")

# =========================================================================
# 1. 데이터 불러오기 및 X, y 세팅
# =========================================================================
df = pd.read_csv('../data/블루리본_최최종_마참내_v1.1.1.csv', encoding='cp949')

# 💡 계산된 베이지안 컬럼(df_koelectra)을 매장명 기준으로 df에 병합
df = pd.merge(df, df_koelectra[bayes_cols_to_keep], on='매장명', how='left')
print(f"ℹ️ 베이지안 변수 병합 후 컬럼 수: {len(df.columns)}")

# df.columns
y = df['블루리본 여부']

# 이미지 모델 결과 컬럼들의 0 값을 NaN으로 변경 (LightGBM 특화 처리)
image_cols = ['고급성_평균', '쾌적성_평균', '감성_평균']
df[image_cols] = df[image_cols].replace(0, np.nan)
print("ℹ️ 이미지 변수 내 0 -> NaN 변환 완료!")

# 'has_missing' 이라는 새로운 파생 변수 생성
# 결측값이 하나라도 있으면 1, 완전한 데이터면 0으로 채워집니다.
df['has_missing'] = df.isnull().any(axis=1).astype(int)

# 💡 결과가 잘 들어갔는지 확인해보기
print(df['has_missing'].value_counts())

# '카테고리' 컬럼을 원-핫 인코딩합니다.
# dtype=int를 넣어야 True/False가 아닌 1/0 형태로 깔끔하게 들어갑니다.
df = pd.get_dummies(df, columns=['카테고리'], prefix='카테고리', dtype=int)


# --- 변수 제거 설정 ---
# 기본적으로 제거할 변수
base_drop_cols = ['매장명', '블루리본 여부', 'idx_family_weighted_score',
                 'total_review_count', '통과된_내부사진_개수', '총_사진_개수']
base_score_cols = [col for col in df.columns if col.endswith('_score')]
base_ratio_cols = [col for col in df.columns if col.endswith('_ratio')]
base_max_cols = [col for col in df.columns if col.endswith('_최대')]
base_mid_cols = [col for col in df.columns if col.endswith('_중앙값')]

base_drop_cols = base_drop_cols + base_score_cols + base_ratio_cols + base_max_cols + base_mid_cols
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

        # 1. 모델 복잡도 (논리적 정렬 및 상한선 축소)
        "n_estimators": trial.suggest_int("n_estimators", 150, 400), # 불필요한 반복 학습 차단
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 5), # 깊이를 보수적으로 제한
        "num_leaves": trial.suggest_int("num_leaves", 10, 20), # 💡 max_depth에 맞춰 최대 15로 제한

        # 2. 과적합 방지용 샘플링
        "subsample": trial.suggest_float("subsample", 0.65, 0.85),
        "subsample_freq": trial.suggest_int("subsample_freq", 1, 3),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 0.80),

        # 3. 리프 최소 데이터 수 (하한선 상향)
        "min_child_samples": trial.suggest_int("min_child_samples", 30, 50), # 💡 조금 더 깐깐하게
        "min_child_weight": trial.suggest_float("min_child_weight", 0.1, 5.0, log=True),

        # 4. 규제 (현재 잘 작동하고 있으므로 범위만 최적화)
        "reg_alpha": trial.suggest_float("reg_alpha", 1.0, 4.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 4.0),

        # 5. 분할 이득 허들
        "min_split_gain": trial.suggest_float("min_split_gain", 0.1, 0.5)
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
# 4. 최적 파라미터로 최종 모델 학습
# =========================================================================
print("\n[ 🚀 최종 모델 학습 시작 ]")

final_model = lgb.LGBMClassifier(
    **best_params, # Optuna에서 찾은 최적 파라미터
    objective="binary",
    metric="binary_logloss",
    boosting_type="gbdt",
    random_state=42,
    verbose=-1,
    n_jobs=-1
)

final_model.fit(X_train, y_train)

# 확률값 계산 (AUC 계산 및 Threshold 조정에 공통으로 사용)
y_train_proba = final_model.predict_proba(X_train)[:, 1]
y_test_proba = final_model.predict_proba(X_test)[:, 1]

# =========================================================================
# 5. 모델 기본 체력 및 과적합 검증 (AUC 중심)
# =========================================================================
train_auc = roc_auc_score(y_train, y_train_proba)
test_auc = roc_auc_score(y_test, y_test_proba)
auc_gap = abs(train_auc - test_auc)

# 기본 임계값(0.5) 기준의 정확도
train_acc_default = accuracy_score(y_train, (y_train_proba >= 0.5).astype(int))
test_acc_default = accuracy_score(y_test, (y_test_proba >= 0.5).astype(int))

print("\n[ 🎯 Train vs Test 과적합 검증 (Default Threshold: 0.5) ]")
print(f"Train ROC-AUC : {train_auc:.4f}")
print(f"Test ROC-AUC  : {test_auc:.4f}")
print(f"AUC Gap       : {auc_gap:.4f} (0.05 이하 권장)")
print(f"Train Accuracy: {train_acc_default:.4f}")
print(f"Test Accuracy : {test_acc_default:.4f}")

# =========================================================================
# 6. 비즈니스 최적화를 위한 Threshold 탐색
# =========================================================================
print("\n[ 🔍 다양한 Threshold에 따른 실전 성능 변화 테스트 ]")
print("Threshold | Accuracy | Precision | Recall | F1-Score")
print("-" * 55)

for threshold in np.arange(0.50, 0.72, 0.02):
    temp_pred = (y_test_proba >= threshold).astype(int)

    acc = accuracy_score(y_test, temp_pred)
    prec = precision_score(y_test, temp_pred, zero_division=0)
    rec = recall_score(y_test, temp_pred)
    f1 = f1_score(y_test, temp_pred)

    print(f"   {threshold:.2f}   |  {acc:.4f}  |   {prec:.4f}  | {rec:.4f} |  {f1:.4f}")


# =========================================================================
# 7. 최종 Threshold 적용 및 상세 리포트 출력
# =========================================================================
# 💡 위 루프 결과를 확인한 후, 가장 마음에 드는 숫자를 아래에 입력하세요!
FINAL_THRESHOLD = 0.52

# Train과 Test 각각에 새로운 Threshold 적용
final_train_pred = (y_train_proba >= FINAL_THRESHOLD).astype(int)
final_test_pred = (y_test_proba >= FINAL_THRESHOLD).astype(int)

print(f"\n[ 📊 최종 실전 성능 (적용된 Threshold: {FINAL_THRESHOLD}) ]")
# 💡 최종 Train 정확도 출력 추가
print(f"최종 Train Accuracy : {accuracy_score(y_train, final_train_pred):.4f}")
print(f"최종 Test Accuracy  : {accuracy_score(y_test, final_test_pred):.4f}")
# 최종 Accuracy Gap 확인 (선택 사항)
print(f"Accuracy Gap        : {abs(accuracy_score(y_train, final_train_pred) - accuracy_score(y_test, final_test_pred)):.4f}")
print("-" * 55)
print(classification_report(y_test, final_test_pred, zero_division=0))


# =========================================================================
# 8. 특성 중요도 (Feature Importance) 확인
# =========================================================================
importance_gain = final_model.booster_.feature_importance(importance_type='gain')

importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': importance_gain
}).sort_values('Importance', ascending=False).head(15)

# print("\n[ 🔥 상위 15개 중요 변수 ]")
# print(importance_df.to_string(index=False))

# 시각화
fig, ax = plt.subplots(figsize=(12, 8))
lgb.plot_importance(final_model, ax=ax, max_num_features=20, height=0.8,
                    importance_type='gain', title='LightGBM Feature Importance (Gain)')
plt.tight_layout()
plt.show()
