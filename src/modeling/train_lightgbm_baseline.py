"""
Description:
    리뷰·이미지·카테고리 특징으로 블루리본 선정 여부를 예측하는 LightGBM 기준 모델입니다.

Author:
    김동혁
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 깨짐 방지 설정
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False


# =================================================
# [공통] 데이터 로드 및 피처/타겟 분리
# =================================================
df = pd.read_csv('../data/블루리본_최최종_마참내_v1.1.csv', encoding='cp949')
y = df['블루리본 여부']


# =================================================
# 1-1. 가중치 점수 아닌 경우 (사용 시 주석 해제)
# =================================================
# weighted_cols = [col for col in df.columns if col.endswith('_weighted_score')]
# drop_cols = ['카테고리', '매장명', '블루리본 여부', '총_사진_개수', '통과된_내부사진_개수'] + weighted_cols
# X_base = df.drop(columns=drop_cols)
# print(len(X_base.columns))

# print(X_base.columns)    # 확인용


# =================================================
# 1-2. 가중치 점수인 경우 (사용 시 주석 해제)
# =================================================
# '_weighted_score'는 살리고, 순수하게 '_score'로 끝나는 컬럼만 추출
base_score_cols = [
    col for col in df.columns
    if col.endswith('_score') and not col.endswith('_weighted_score')
]
drop_cols = ['카테고리', '매장명', '블루리본 여부', '총_사진_개수', '통과된_내부사진_개수'] + base_score_cols
X_base = df.drop(columns=drop_cols)

print(X_base.columns)    # 확인용


# =================================================
# [공통] Train / Test 데이터셋 분리 (공정성 유지용)
# =================================================
# random_state, test_size, stratify는 세 명 모두 무조건 이대로 고정해야 합니다!
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_base, y, test_size=0.2, random_state=42, stratify=y
)

print(f"공통 분리 완료 - Train: {X_train_raw.shape}, Test: {X_test_raw.shape}")
# => train: (1152, 49), test: (288, 49)
X_train = X_train_raw.copy()
X_test = X_test_raw.copy()

# 수치형, 결측치 등 정보 확인
print(X_train.info())


# =========================================================================
# 2. 이미지 모델 결과 컬럼들의 0 값을 NaN으로 변경 (LGBM)
# =========================================================================
image_cols = [
    '고급성_평균', '고급성_중앙값', '고급성_최대',
    '쾌적성_평균', '쾌적성_중앙값', '쾌적성_최대',
    '감성_평균', '감성_중앙값', '감성_최대'
]

# Train셋과 Test셋 모두 동일하게 0을 NaN으로 대체합니다.
X_train[image_cols] = X_train[image_cols].replace(0, np.nan)
X_test[image_cols] = X_test[image_cols].replace(0, np.nan)

print("ℹ️ 이미지 변수 내 0 -> NaN 변환 완료!")


# =========================================================================
# 3. LightGBM 모델 선언 및 학습
# =========================================================================
# LightGBM은 자체적으로 NaN(결측치)을 최적의 방향으로 분류하는 알고리즘을 가집니다.
model = lgb.LGBMClassifier(
    random_state=42,
    verbose=-1,
    n_estimators=100,      # 기본 트리 개수
    learning_rate=0.05     # 학습률 (필요시 조정 가능)
)

model.fit(X_train, y_train)


# =========================================================================
# 4. 예측 및 최종 성능 평가
# =========================================================================
# Train 데이터 예측
y_train_pred = model.predict(X_train)
y_train_pred_proba = model.predict_proba(X_train)[:, 1]

# Test 데이터 예측
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print(f"\n{'='*50}")
# '블루리본 여부'가 타겟이므로 정확도와 함께 불균형 데이터에 강한 ROC AUC를 메인으로 확인합니다.
print(f"📊 Train 정확도 (Accuracy) : {accuracy_score(y_train, y_train_pred):.4f}")
print(f"🏆 Test  정확도 (Accuracy) : {accuracy_score(y_test, y_pred):.4f}")
print("-" * 50)
print(f"📊 Train ROC AUC 점수  : {roc_auc_score(y_train, y_train_pred_proba):.4f}")
print(f"🏆 Test  ROC AUC 점수  : {roc_auc_score(y_test, y_pred_proba):.4f}")
print(f"{'='*50}\n")
print("[ 분류 성과 리포트 ]")
print(classification_report(y_test, y_pred, zero_division=0))


# =========================================================================
# 5. 특성 중요도 (Feature Importance) 확인
# =========================================================================
# LightGBM의 특성 중요도는 'split'(분할 횟수)과 'gain'(분할로 얻은 정보 이득량) 두 가지로 확인할 수 있습니다.
importance_split = model.booster_.feature_importance(importance_type='split')
importance_gain = model.booster_.feature_importance(importance_type='gain')

importance_df_split = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance (Split)': importance_split
}).sort_values('Importance (Split)', ascending=False).head(15)

importance_df_gain = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance (Gain)': importance_gain
}).sort_values('Importance (Gain)', ascending=False).head(15)

print("\n[ 🔥 상위 15개 중요 변수 (Split 기준: 변수가 분할에 사용된 횟수) ]")
print(importance_df_split.to_string(index=False))

print("\n[ 🌟 상위 15개 중요 변수 (Gain 기준: 변수 분할로 인한 정확도 향상 기여도) ]")
print(importance_df_gain.to_string(index=False))


# =========================================================================
# 6. 변수 간 상관관계 분석 (Correlation Analysis)
# =========================================================================
print("\n[ 🔍 변수 간 상관관계 분석 ]")
# 수치형 변수만을 대상으로 피어슨 상관계수 행렬 계산
corr_matrix = X_train.corr(numeric_only=True)

# 하삼각행렬을 가리는 마스크 생성 (대각선 아래쪽을 True로 만들어 히트맵에서 숨김)
mask = np.tril(np.ones_like(corr_matrix, dtype=bool), k=-1)

# 1. 상관관계 히트맵 시각화 및 저장 (상삼각행렬만 표시)
plt.figure(figsize=(16, 12))
sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('독립 변수 간 상관관계 히트맵 (Train Data)', fontsize=16)
plt.tight_layout()
plt.show()
# plt.savefig('../data/feature_correlation_heatmap.png')
# plt.close()
# print("   ✅ 상관관계 히트맵이 '../data/feature_correlation_heatmap.png'로 저장되었습니다.")

# 2. 강한 상관관계를 가지는 변수 쌍 추출 (절댓값 0.7 이상)
# 자기 자신 및 중복 비교를 제외하기 위해 상삼각행렬(Upper Triangle) 사용
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr = [(col1, col2, upper_tri.loc[col1, col2])
             for col1 in upper_tri.columns for col2 in upper_tri.columns
             if abs(upper_tri.loc[col1, col2]) >= 0.7]

if high_corr:
    print("\n🚨 [강한 상관관계 (절댓값 0.7 이상) 변수 쌍]")
    high_corr_df = pd.DataFrame(high_corr, columns=['변수1', '변수2', '상관계수'])
    print(len(high_corr_df), "쌍 발견")
    high_corr_df = high_corr_df.sort_values(by='상관계수', key=abs, ascending=False)
    print(high_corr_df.to_string(index=False))
else:
    print("\n💡 강한 상관관계(절댓값 0.7 이상)를 가지는 변수 쌍이 없습니다.")


# =========================================================================
# 7. LightGBM 트리 구조 시각화 (Tree Structure)
# =========================================================================
# [Graphviz 환경 변수 설정]
# Graphviz가 설치된 경로의 bin 폴더를 지정합니다. (기본 설치 경로가 다를 경우 수정 필요)
import os
graphviz_path = r'C:\Program Files\Graphviz\bin'
if graphviz_path not in os.environ["PATH"]:
    os.environ["PATH"] += os.pathsep + graphviz_path

print("\n[ 🌳 LightGBM 트리 구조 시각화 ]")
try:
    fig, ax = plt.subplots(figsize=(20, 10))
    # 첫 번째 트리(tree_index=0) 시각화
    lgb.plot_tree(model, tree_index=0, ax=ax,
                  show_info=['split_gain', 'internal_value', 'internal_count', 'leaf_count'])
    plt.title('LightGBM Model Tree Structure (Tree Index: 0)', fontsize=16)
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"🚨 트리 시각화 실패: {e}")
    print("💡 Graphviz 패키지가 설치되어 있어야 트리 시각화가 가능합니다.")
    print("설치 방법: pip install graphviz 및 OS별 Graphviz 프로그램(실행 파일) 설치 필요")
