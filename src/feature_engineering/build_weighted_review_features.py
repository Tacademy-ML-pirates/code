"""
Description:
    키워드별 리뷰 감성 점수에 언급 빈도와 통계 검정 결과를 반영해 가중 특징을 생성합니다.

Author:
    김동혁
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# 경고 메시지 무시 및 한글 폰트 설정
warnings.filterwarnings('ignore')
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False


def calculate_derived_metrics(df: pd.DataFrame, raw_file_path: str) -> tuple[pd.DataFrame, list, list, list]:
    """매장별 총 리뷰 수를 기반으로 특정 지수의 언급 비율(Ratio) 및 가중 점수(Weighted Score)를 계산합니다."""
    print("1. 매장별 특정 지수 언급 비율(Ratio) 및 가중 점수(Weighted Score)를 계산합니다...")
    print("   원본 리뷰 데이터를 읽어 '진짜 총 리뷰 수'를 계산 중...")

    df_raw = pd.read_excel(raw_file_path)
    df_valid = df_raw[(df_raw['리뷰 내용'].notna()) & (df_raw['리뷰 내용'] != '텍스트 리뷰 없음')]
    total_reviews = df_valid.groupby('매장명').size().reset_index(name='total_review_count')

    # 기존에 'total_review_count' 컬럼이 있다면 삭제하여 merge 시 _x, _y가 생기는 것을 방지
    if 'total_review_count' in df.columns:
        df = df.drop(columns=['total_review_count'])

    df = pd.merge(df, total_reviews, on='매장명', how='left')
    df['total_review_count'] = df['total_review_count'].fillna(0)

    count_cols = [col for col in df.columns if col.endswith('_count') and col != 'total_review_count']
    ratio_cols = []
    weighted_score_cols = []

    for count_col in count_cols:
        base_name = count_col.replace('_count', '')
        score_col = base_name + '_score'
        ratio_col = base_name + '_ratio'
        weighted_score_col = base_name + '_weighted_score'

        # 비율(Ratio) 계산
        df[ratio_col] = np.where(df['total_review_count'] > 0, df[count_col] / df['total_review_count'], 0)
        ratio_cols.append(ratio_col)

        # 가중 점수(Weighted Score) 계산: Score * Ratio
        df[weighted_score_col] = df[score_col] * df[ratio_col]
        weighted_score_cols.append(weighted_score_col)

    return df, count_cols, ratio_cols, weighted_score_cols


def perform_ttest(df: pd.DataFrame, category_col: str, blue_ribbon_col: str, count_cols: list) -> pd.DataFrame:
    """전체 및 카테고리별로 T-test를 수행하여 결과를 반환합니다."""
    print("\n2. 블루리본 여부에 따른 [점수 / 빈도 / 비율 / 가중 점수] T-test 검정 (전체 및 카테고리별)")

    test_results = []
    categories = ['전체'] + [cat for cat in df[category_col].unique() if pd.notna(cat)] if category_col else ['전체']

    for cat in categories:
        target_df = df if cat == '전체' else df[df[category_col] == cat]

        blue = target_df[target_df[blue_ribbon_col] == 1]
        non_blue = target_df[target_df[blue_ribbon_col] == 0]

        if len(blue) < 2 or len(non_blue) < 2:
            continue

        for count_col in count_cols:
            base_name = count_col.replace('_count', '')
            score_col = base_name + '_score'
            ratio_col = base_name + '_ratio'
            weighted_score_col = base_name + '_weighted_score'
            idx_display = base_name.replace('idx_', '')

            _, p_score = stats.ttest_ind(blue[score_col], non_blue[score_col], equal_var=False, nan_policy='omit')
            _, p_count = stats.ttest_ind(blue[count_col], non_blue[count_col], equal_var=False, nan_policy='omit')
            _, p_ratio = stats.ttest_ind(blue[ratio_col], non_blue[ratio_col], equal_var=False, nan_policy='omit')
            _, p_weighted = stats.ttest_ind(blue[weighted_score_col], non_blue[weighted_score_col], equal_var=False, nan_policy='omit')

            test_results.append({
                '분석 기준': cat,
                '지수': idx_display,
                '점수 P-value': f"{p_score:.4f}" if not np.isnan(p_score) else "NaN",
                '빈도 P-value': f"{p_count:.4f}" if not np.isnan(p_count) else "NaN",
                '비율 P-value': f"{p_ratio:.4f}" if not np.isnan(p_ratio) else "NaN",
                '가중점수 P-value': f"{p_weighted:.4f}" if not np.isnan(p_weighted) else "NaN"
            })

    return pd.DataFrame(test_results)


def create_visualizations(df: pd.DataFrame, category_col: str, blue_ribbon_col: str, count_cols: list, ratio_cols: list, weighted_score_cols: list):
    """점수, 빈도, 비율, 가중 점수에 대한 시각화 차트를 전체/카테고리별로 생성합니다."""
    print("\n3. 블루리본 여부별 점수/빈도/비율/가중점수 시각화 차트 4종을 생성합니다...")

    metrics = {
        'score': ([col for col in df.columns if col.endswith('_score') and not col.endswith('_weighted_score')], '평균 긍정 확률 점수'),
        'count': (count_cols, '평균 언급 횟수 (Count)'),
        'ratio': (ratio_cols, '평균 언급 비율 (Ratio)'),
        'weighted_score': (weighted_score_cols, '평균 가중 점수 (Weighted Score)')
    }

    for metric_name, (cols, y_label) in metrics.items():
        df_melt = df.melt(id_vars=[category_col, blue_ribbon_col], value_vars=cols, var_name='지수', value_name='값')
        df_melt['지수'] = df_melt['지수'].str.replace('idx_', '').str.replace(f'_{metric_name}', '')

        # 1. 전체 비교 차트
        plt.figure(figsize=(14, 7))
        sns.barplot(data=df_melt, x='지수', y='값', hue=blue_ribbon_col, errorbar=None, palette='Set2')
        plt.title(f'블루리본 여부에 따른 지수별 "{metric_name.capitalize()}" 비교 (전체)', fontsize=16)
        plt.xlabel('평가 지수', fontsize=12)
        plt.ylabel(y_label, fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'../data/blue_ribbon_{metric_name}_comparison_overall.png')
        plt.close()

        # 2. 카테고리별 비교 차트
        g = sns.catplot(data=df_melt, x='지수', y='값', hue=blue_ribbon_col,
                        col=category_col, col_wrap=2, kind='bar',
                        height=4, aspect=1.5, errorbar=None, palette='Set2', sharex=True)
        g.fig.suptitle(f'카테고리별 블루리본 여부 "{metric_name.capitalize()}" 차이 비교', fontsize=18, y=1.03)
        g.set_axis_labels("", y_label)

        for ax in g.axes.flatten():
            ax.tick_params(axis='x', labelrotation=45)

        plt.tight_layout()
        plt.savefig(f'../data/blue_ribbon_{metric_name}_comparison_category.png')
        plt.close()

    print("   ✅ 전체 및 카테고리별 시각화 이미지(총 8장)가 ../data/ 폴더에 저장되었습니다.")


def main():
    feature_path = '../data/KoELECTRA_X_feature_v2.1.csv'  # review_senti_v2.0.py에서 count까지 계산된 파일
    raw_path = '../data/merge_blue.xlsx'
    output_csv = '../data/KoELECTRA_X_feature_weighted.csv'
    ttest_output = '../data/T_test_results_detailed.csv'

    print("데이터를 불러옵니다...")
    df = pd.read_csv(feature_path, encoding='utf-8-sig')

    blue_ribbon_col = next((col for col in df.columns if '블루리본 여부' in col), None)
    category_col = next((col for col in df.columns if '카테고리' in col), None)

    if not blue_ribbon_col:
        raise ValueError("블루리본 여부 컬럼을 찾을 수 없습니다.")

    # Step 1. 파생 변수(비율 및 가중 점수) 계산
    df, count_cols, ratio_cols, weighted_score_cols = calculate_derived_metrics(df, raw_path)

    # Step 2. T-test 검정 및 결과 저장
    # stats_df = perform_ttest(df, category_col, blue_ribbon_col, count_cols)
    # stats_df.to_csv(ttest_output, encoding='utf-8-sig', index=False)
    # print(f"   ✅ 카테고리별 상세 결과가 '{ttest_output}'에 저장되었습니다!")
    # print("\n[전체 식당 기준 T-test 요약]")
    # print(stats_df[stats_df['분석 기준'] == '전체'].to_string(index=False))

    # # Step 3. 시각화
    # create_visualizations(df, category_col, blue_ribbon_col, count_cols, ratio_cols, weighted_score_cols)

    # Step 4. 최종 데이터 저장
    df.to_csv(output_csv, encoding='utf-8-sig', index=False)
    print(f"\n✅ 가중 점수(Weighted Score) 변수가 추가된 최종 데이터가 '{output_csv}'로 저장되었습니다!")


if __name__ == "__main__":
    main()
