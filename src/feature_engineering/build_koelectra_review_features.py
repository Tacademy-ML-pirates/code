"""
Description:
    리뷰 문장에서 주요 키워드 주변 문맥을 추출하고 KoELECTRA 감성 점수를 생성합니다.

Author:
    김동혁
"""

import re

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ==========================================
# 환경 설정 및 상수 정의
# ==========================================
KEYWORD_DICT = {
    'idx_family': r'가족|부모님|어머니|아버지|엄마|아빠|아이|아기|애기|자녀|아들|딸|할머니|할아버지|식구|모시고|가족모임|가족 단위|가족단위|어버이날|환갑|칠순',
    'idx_anniversary': r'생일|생신|기념일|결혼기념일|프로포즈|anniversary|축하|파티|졸업|합격|크리스마스|연말|특별한 날|특별한날|어버이날|졸업식|환갑|칠순',
    'idx_date': r'데이트|커플|남자친구|여자친구|남친|여친|연인|소개팅|썸|오붓',
    'idx_distance': r'멀리서|일부러|찾아가|찾아온|찾아왔|원정|여행|관광|출장|들렀|방문하려고|예약하고 방문|서울에서|지방에서',
    'idx_revisit': r'재방문|또 갈|또갈|또 가|또가|다시 방문|다시 가|다시가|다음에 또|다음에도|또 오|또오|단골|재방문의사|재방문할',
    'idx_service': r'친절|서비스|응대|직원|서버|설명|배려|접객|안내|셰프님|쉐프님|사장님|챙겨',
    'idx_ambiance': r'분위기|인테리어|공간|조명|음악|감성|아늑|깔끔한 분위기|예쁜|예뻐|멋진|고급스러운|고급진|쾌적',
    'idx_view': r'뷰|전망|야경|창가|통창|바다|한강|남산|풍경|정원|테라스|루프탑|오션뷰|리버뷰',
    'idx_taste': r'맛있|맛이|맛도|풍미|식감|재료|신선|간이|부드러|고소|육즙|향이|진한|담백|퀄리티|조리|소스|훌륭|좋았',
    'idx_price': r'가격|가성비|비싸|저렴|합리|금액|비용|돈값|가격대|값어치|가심비',
    'idx_waiting': r'웨이팅|대기|줄서|줄 서|예약|캐치테이블|예약필수|오픈런|기다렸|기다림',
    'idx_luxury': r'코스|파인다이닝|오마카세|디테일|특별|고급|미슐랭|미쉐린|럭셔리|프라이빗|대접|기분내기|격식|섬세',
    'idx_disappoint': r'아쉬|별로|불친절|실망|비추|최악|느려|늦게|문제'
}

MODEL_NAME = "Copycats/koelectra-base-v3-generalized-sentiment-analysis"
BATCH_SIZE = 32
MAX_LENGTH = 128
CONTEXT_WINDOW = 15


def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """리뷰 데이터를 불러오고 결측치 및 불필요한 데이터를 제거합니다."""
    print("데이터를 불러옵니다...")
    df_raw = pd.read_excel(file_path)

    # 불필요한 컬럼 제거
    cols_to_drop = [col for col in ['Unnamed: 0', '별점'] if col in df_raw.columns]
    if cols_to_drop:
        df_raw.drop(columns=cols_to_drop, inplace=True)

    print(f'결측값 및 "텍스트 리뷰 없음" 데이터 제거 전 데이터 수: {len(df_raw)}')

    # 유효한 리뷰 데이터만 필터링
    df_valid = df_raw[df_raw['리뷰 내용'].notna()].copy()
    df_valid = df_valid[df_valid['리뷰 내용'] != '텍스트 리뷰 없음']

    print(f'결측값 및 "텍스트 리뷰 없음" 데이터 제거 후 데이터 수: {len(df_valid)}')
    return df_valid


def extract_sentences_and_match_keywords(df: pd.DataFrame) -> pd.DataFrame:
    """리뷰를 문장 단위로 분리하고 키워드를 매칭합니다."""
    print("\n1. 리뷰를 문장 단위로 분리하고 키워드를 매칭합니다...")

    # 리뷰를 마침표, 느낌표, 줄바꿈 등으로 분리하여 새로운 행 생성
    df_sentences = df.assign(sentence=df['리뷰 내용'].str.split(r'[.!?\n]+')).explode('sentence')
    df_sentences['sentence'] = df_sentences['sentence'].str.strip()
    df_sentences = df_sentences[df_sentences['sentence'] != ""]

    # 각 문장이 13개 지수 중 어디에 속하는지 검사
    for idx_name, keywords in KEYWORD_DICT.items():
        df_sentences[f'has_{idx_name}'] = df_sentences['sentence'].str.contains(keywords, regex=True, na=False)

    # 적어도 하나의 지수에 속하는 문장만 필터링 (연산량 최적화)
    index_columns = [f'has_{k}' for k in KEYWORD_DICT.keys()]
    df_target = df_sentences[df_sentences[index_columns].any(axis=1)].copy()

    return df_target


def extract_contexts(df_target: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """정규표현식을 사용하여 키워드 주변 문맥을 추출합니다."""
    compiled_dict = {k: re.compile(v) for k, v in KEYWORD_DICT.items()}

    for idx_name in KEYWORD_DICT.keys():
        df_target[f'{idx_name}_score'] = np.nan

    inference_tasks = []

    print("⏳ 키워드 주변 문맥 추출 중 (고속 모드)...")
    for row in tqdm(df_target.itertuples(), total=len(df_target)):
        text = str(row.sentence)
        index = row.Index

        for idx_name, compiled_re in compiled_dict.items():
            col_has = f'has_{idx_name}'
            col_score = f'{idx_name}_score'

            if getattr(row, col_has):
                match = compiled_re.search(text)
                if match:
                    start, end = match.start(), match.end()
                    sub_start = max(0, start - CONTEXT_WINDOW)
                    sub_end = min(len(text), end + CONTEXT_WINDOW)
                    sub_text = text[sub_start:sub_end]
                else:
                    sub_text = text

                inference_tasks.append((index, col_score, sub_text))

    return df_target, inference_tasks


def predict_sentiment(inference_tasks: list) -> list:
    """KoELECTRA 모델을 사용하여 문맥의 감성을 분석합니다."""
    print("\n2. 모델 및 토크나이저 로드 중 (CPU)...")
    device = torch.device('cpu')

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    print(f"🚀 총 {len(inference_tasks)}개의 세부 문맥 CPU 배치 추론 시작...")
    results = []

    for i in tqdm(range(0, len(inference_tasks), BATCH_SIZE)):
        batch_tasks = inference_tasks[i : i + BATCH_SIZE]
        batch_texts = [task[2] for task in batch_tasks]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # 긍정 확률(1) 추출
        probs = F.softmax(outputs.logits, dim=-1)[:, 1].tolist()

        for task, prob in zip(batch_tasks, probs):
            results.append((task[0], task[1], prob))

    return results


def aggregate_features(df_target: pd.DataFrame, df_raw: pd.DataFrame) -> pd.DataFrame:
    """매장별로 감성 확률 점수와 언급 빈도를 집계하여 병합합니다."""
    print("\n3. 매장별 최종 X 변수를 생성합니다...")

    # 동일 리뷰(인덱스)의 중복을 제거하여 1리뷰 1점수로 변환
    df_target_unique = df_target[~df_target.index.duplicated(keep='first')]
    final_features = []

    for rest_name, group in df_target_unique.groupby('매장명'):
        feature_row = {'매장명': rest_name}

        for idx_name in KEYWORD_DICT.keys():
            col_score = f'{idx_name}_score'

            mean_score = group[col_score].mean()
            mention_count = group[col_score].count()

            feature_row[col_score] = 0.0 if pd.isna(mean_score) else mean_score
            feature_row[f'{idx_name}_count'] = mention_count

        final_features.append(feature_row)

    df_final = pd.DataFrame(final_features)

    print("4. 매장 카테고리 및 블루리본 여부 정보를 병합합니다...")
    info_cols = ['매장명']
    for col in df_raw.columns:
        if '카테고리' in col or '블루리본 여부' in col:
            if col not in info_cols:
                info_cols.append(col)

    df_info = df_raw[info_cols].drop_duplicates(subset=['매장명'])
    df_final = pd.merge(df_final, df_info, on='매장명', how='left')

    # 컬럼 순서 정렬
    score_cols = [col for col in df_final.columns if col.endswith('_score')]
    count_cols = [col for col in df_final.columns if col.endswith('_count')]
    front_cols = [col for col in df_final.columns if col not in score_cols and col not in count_cols]

    return df_final[front_cols + score_cols + count_cols]


def main():
    file_path = '../data/merge_blue.xlsx'
    output_path = '../data/KoELECTRA_X_feature.csv'

    # 1. 데이터 로드 및 전처리
    df_valid = load_and_clean_data(file_path)

    # 2. 문장 분리 및 키워드 매칭
    df_target = extract_sentences_and_match_keywords(df_valid)

    # 3. 문맥 추출
    df_target, inference_tasks = extract_contexts(df_target)

    # 4. 감성 분석 추론
    results = predict_sentiment(inference_tasks)

    print("\n🧩 계산된 점수를 데이터프레임에 매칭 중...")
    for index, col_score, prob in results:
        df_target.at[index, col_score] = prob

    # 불필요한 has_ 컬럼 삭제
    cols_to_drop = [col for col in df_target.columns if col.startswith('has_')]
    df_target.drop(columns=cols_to_drop, inplace=True)
    print("✅ 리뷰별/지수별 세부 감성 확률 스코어 매칭 및 정리 완료!")

    # 5. 최종 데이터 집계 및 저장
    df_final_X = aggregate_features(df_target, df_valid)

    print("\n🔥 세부 긍정 확률 스코어가 반영된 최종 데이터프레임 완성!")
    print(df_final_X.head())
    print(f"데이터 크기: {df_final_X.shape}")

    df_final_X.to_csv(output_path, encoding='utf-8-sig', index=False)
    print(f"✅ 최종 데이터가 '{output_path}'에 저장되었습니다.")


if __name__ == "__main__":
    main()
