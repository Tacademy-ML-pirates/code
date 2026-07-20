"""
Description:
    LightGBM 예측 결과와 SHAP 값을 이용해 식당별 강점, 약점, 업종 평균 비교 리포트를 생성합니다.

Author:
    이용현

Source:
    Notion ML database, page 37dbf34c-c2ef-805b-ae79-d82faab62bc7
"""

import pandas as pd
import numpy as np
import shap
import warnings
warnings.filterwarnings("ignore")

print("\n=======================================================")
print("🚀 [실전 모드] 컬럼명 매칭 + LightGBM 블루리본 예측 & SHAP 컨설팅")
print("=======================================================")

# 1. 파일 불러오기 (실제 파일명 확인 필요)
df_new = pd.read_csv("/content/식당데이터_test_찐최종합본.csv")

# 🚨 [컬럼명 자동 매칭 마법]
rename_dict = {}
for col in df_new.columns:
    if col.endswith('_score') and not col.endswith('_weighted_score'):
        rename_dict[col] = col.replace('_score', '_weighted_score')

if rename_dict:
    df_new = df_new.rename(columns=rename_dict)
    print(f"🔄 컬럼명 변환 완료: {len(rename_dict)}개의 점수 컬럼을 모델용 이름으로 매칭했습니다.")

# 매장명 따로 보관
new_names = df_new["매장명"].tolist() if "매장명" in df_new.columns else [f"매장_{i}" for i in range(len(df_new))]

# ⭐ [추가 1] 카테고리별 평균값 미리 계산 및 저장 (원핫인코딩 전)
store_categories = []
category_means = {}

if "카테고리" in df_new.columns:
    store_categories = df_new["카테고리"].tolist()
    # 숫자형 컬럼에 대해서만 카테고리별 평균 계산
    category_means = df_new.groupby("카테고리").mean(numeric_only=True).to_dict('index')
else:
    store_categories = ["알수없음"] * len(df_new)

# 2. 전처리 (카테고리 원핫인코딩)
if "카테고리" in df_new.columns:
    df_new = pd.get_dummies(df_new, columns=["카테고리"], drop_first=False, dtype=int)
    print("✅ 카테고리 원핫인코딩 완료!")

# 이미지 내부 점수 결측치 처리 (학습 환경과 동일하게 0을 NaN으로)
img_cols = ["고급성_평균", "고급성_중앙값", "고급성_최대", "쾌적성_평균", "쾌적성_중앙값", "쾌적성_최대", "감성_평균", "감성_중앙값", "감성_최대"]
img_cols = [c for c in img_cols if c in df_new.columns]
df_new[img_cols] = df_new[img_cols].replace(0, np.nan)

# 🚨 학습 데이터 컬럼과 구조 완벽 동기화 (final_model 기준)
X_new = df_new.reindex(columns=final_model.feature_name_, fill_value=0)

# 🔍 [데이터 안전 검사]
score_sum = X_new.filter(like='weighted_score').sum().sum()
if score_sum == 0:
    print("⚠️ 경고: 점수 데이터가 0입니다. X_train.columns와 파일의 컬럼명을 대조해야 합니다.")
else:
    print("🎯 데이터 동기화 성공! 점수 데이터가 모델에 정상적으로 주입되었습니다.")

# =====================================================
# 3. LightGBM 단일 모델로 합격 확률 및 결과 예측
# =====================================================
# 학습 단계에서 찾은 최적의 Threshold 적용 (예: 0.52)
th = 0.542

# LightGBM 예측 확률 가져오기
new_prob = final_model.predict_proba(X_new)[:, 1]
new_pred = (new_prob >= th).astype(int)

result_df = pd.DataFrame({
    "매장명": new_names,
    "합격확률(%)": np.round(new_prob * 100, 1),
    "예측결과": ["🏆블루리본 합격" if p == 1 else "❌탈락 위기" for p in new_pred]
})

print("\n📊 [1단계] 매장별 블루리본 예측 결과 요약")
print(result_df.to_string(index=False))

# =====================================================
# 4. SHAP을 활용한 컨설팅 리포트 자동 생성
# =====================================================
print("\n=======================================================")
print("💡 [2단계] SHAP 기반 매장별 맞춤형 컨설팅 리포트 (동종업계 비교 포함)")
print("=======================================================")

# final_model을 기반으로 SHAP Explainer 생성
explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_new)

# LightGBM 버전에 따라 shap_values가 리스트로 반환될 수 있음 (이진 분류 시 클래스 1에 대한 값 추출)
if isinstance(shap_values, list):
    shap_values_to_use = shap_values[1]
else:
    shap_values_to_use = shap_values

for i in range(len(X_new)):
    store_name = result_df.loc[i, "매장명"]
    prob = result_df.loc[i, "합격확률(%)"]
    pred_res = result_df.loc[i, "예측결과"]
    store_cat = store_categories[i]

    feature_impacts = list(zip(X_new.columns, shap_values_to_use[i]))

    positive_factors = sorted([f for f in feature_impacts if f[1] > 0], key=lambda x: x[1], reverse=True)
    negative_factors = sorted([f for f in feature_impacts if f[1] < 0], key=lambda x: x[1])

    print(f"\n🏪 [{store_name}] 컨설팅 진단: {pred_res} (확률: {prob}%) | 소속 카테고리: {store_cat}")
    print("  ✅ [강점/유지] 이 매장을 돋보이게 하는 핵심 요인:")
    for feat, val in positive_factors[:3]:
        actual_val = X_new.iloc[i][feat]

        # ⭐ [동종업계(카테고리) 평균 비교]
        if store_cat in category_means and feat in category_means[store_cat] and not feat.startswith("카테고리_"):
            mean_val = category_means[store_cat][feat]
            diff = actual_val - mean_val
            diff_text = f" -> 동종업계 평균({mean_val:.2f}) 대비 {abs(diff):.2f}점 {'높음 ⬆️' if diff >= 0 else '부족 ⬇️'}"
        else:
            diff_text = ""

        print(f"      - {feat} (영향력: +{val:.2f} / 현재수치: {actual_val:.3f}){diff_text}")

    print("  \n⚠️ [약점/보완] 반드시 개선해야 할 위험 요인:")
    if len(negative_factors) > 0:
        for feat, val in negative_factors[:3]:
            actual_val = X_new.iloc[i][feat]

            # ⭐ [동종업계(카테고리) 평균 비교]
            if store_cat in category_means and feat in category_means[store_cat] and not feat.startswith("카테고리_"):
                mean_val = category_means[store_cat][feat]
                diff = actual_val - mean_val
                diff_text = f" -> 동종업계 평균({mean_val:.2f}) 대비 {abs(diff):.2f}점 {'높음 ⬆️' if diff >= 0 else '부족 ⬇️'}"
            else:
                diff_text = ""

            print(f"      - {feat} (영향력: {val:.2f} / 현재수치: {actual_val:.3f}){diff_text}")
    else:
        print("      - 특별한 치명적 약점이 발견되지 않았습니다. 훌륭합니다!")

print("\n=======================================================")
