"""
Description:
    가중 리뷰, 이미지, 카테고리 변수를 사용해 과적합을 억제하는 XGBoost 파라미터를 탐색합니다.

Author:
    이용현

Source:
    Notion ML database, page 37cbf34c-c2ef-80bf-a5ba-f37712a8eb8b
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, roc_auc_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

print("\n[ 🚀 XGBoost를 활용한 블루리본 예측 최종 모델링 (과적합 방지 모드) ]")

# 1. 새로운 데이터 불러오기 (cp949 인코딩 적용)
df = pd.read_csv('블루리본_최최종_마참내_v1.1.csv', encoding='cp949')

# 2. X, y 세팅
y = df['블루리본 여부']  # 띄어쓰기 주의

# (1) Weighted 변수 추출 (기존 ratio 대신 weighted 활용)
weighted_cols = [col for col in df.columns if 'weighted' in col]

# (2) 이미지 관련 점수 변수 (결측치 NaN을 그대로 둡니다!)
img_cols = ['고급성_평균', '고급성_중앙값', '고급성_최대',
            '쾌적성_평균', '쾌적성_중앙값', '쾌적성_최대',
            '감성_평균', '감성_중앙값', '감성_최대']
img_cols = [col for col in img_cols if col in df.columns]

# (3) 카테고리 변수 원-핫 인코딩 (One-Hot Encoding)
df_category = pd.get_dummies(df[['카테고리']], drop_first=True)

# 최종 X 합치기 (빈칸은 0이 아닌 원래의 NaN 상태로 유지)
X = pd.concat([df[weighted_cols], df[img_cols], df_category], axis=1)

print(f"✅ 사용된 총 변수 개수: {len(X.columns)}개")

# 3. Train / Test 분리 (비율 유지 stratify)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Optuna를 이용한 과적합 방지 하이퍼파라미터 튜닝
print("\n[ 🛡️ 과적합 방지 모드: Optuna 재튜닝 시작 ]")

def objective_xgb_regularized(trial):
    params = {
        # 1. 모델 복잡도 줄이기
        'n_estimators': trial.suggest_int('n_estimators', 100, 300),
        'max_depth': trial.suggest_int('max_depth', 2, 6),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),

        # 2. 데이터 & 피처 샘플링 (일부만 보고 판단하게 해서 외우는 걸 막음)
        'subsample': trial.suggest_float('subsample', 0.5, 0.8),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.8),

        # 3. 강력한 규제(Regularization) 파라미터 추가
        'gamma': trial.suggest_float('gamma', 0.1, 5.0), # 가지치기 컷오프
        'min_child_weight': trial.suggest_int('min_child_weight', 3, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10.0, log=True), # L1 규제
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0, log=True), # L2 규제

        'random_state': 42,
        'eval_metric': 'logloss'
    }

    xgb_model = xgb.XGBClassifier(**params)
    score = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='roc_auc')
    return np.mean(score)

study_xgb_reg = optuna.create_study(direction='maximize', study_name="XGB_Anti_Overfit")
study_xgb_reg.optimize(objective_xgb_regularized, n_trials=50)

print("\n--- 🏆 과적합 방지 튜닝 완료 ---")
print("최적의 파라미터:", study_xgb_reg.best_params)

# 5. 최적 파라미터로 모델 최종 학습
best_xgb_reg = xgb.XGBClassifier(**study_xgb_reg.best_params, random_state=42, eval_metric='logloss')
best_xgb_reg.fit(X_train, y_train)

# 6. 실전(Test) 데이터 예측 및 과적합 검증
y_test_pred_new = best_xgb_reg.predict(X_test)
y_test_proba_new = best_xgb_reg.predict_proba(X_test)[:, 1]
y_train_proba_new = best_xgb_reg.predict_proba(X_train)[:, 1]

# --- Train vs Test 과적합 검증 ---
train_auc_new = roc_auc_score(y_train, y_train_proba_new)
test_auc_new = roc_auc_score(y_test, y_test_proba_new)

print("\n=== 🎯 Train vs Test 과적합 검증 ===")
print(f"Train ROC-AUC : {train_auc_new:.4f}")
print(f"Test ROC-AUC  : {test_auc_new:.4f}")
print(f"-> 두 지표의 차이: {abs(train_auc_new - test_auc_new):.4f}")

# --- Test 평가지표 출력 ---
print("\n=== 📊 실전(Test) 평가지표 ===")
print(f"정확도 (Accuracy)  : {accuracy_score(y_test, y_test_pred_new):.4f}")
print(f"정밀도 (Precision) : {precision_score(y_test, y_test_pred_new):.4f}")
print(f"재현율 (Recall)    : {recall_score(y_test, y_test_pred_new):.4f}")
print(f"F1 Score         : {f1_score(y_test, y_test_pred_new):.4f}")
print("\n[ 상세 분류 리포트 (0: 비선정, 1: 선정) ]")
print(classification_report(y_test, y_test_pred_new))

# 7. XGBoost 변수 중요도 추출 및 시각화
importance_xgb = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': best_xgb_reg.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\n--- 🏆 과적합 방지 XGBoost 핵심 변수 (상위 10개) ---")
print(importance_xgb.head(10))

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_xgb.head(20), palette='crest')
plt.title("과적합 방지 XGBoost 모델의 핵심 변수 중요도 (Top 20)")
plt.xlabel('중요도 (Importance)')
plt.ylabel('변수명 (Feature)')
plt.tight_layout()
plt.show()
