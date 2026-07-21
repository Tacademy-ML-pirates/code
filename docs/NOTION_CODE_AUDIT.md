# Notion 코드 가져오기 결과

Notion `ML` 데이터베이스의 코드와 기존 로컬 Python 및 Jupyter 코드를 비교한 결과입니다.

## 비교 기준

- Python 주석과 공백을 제거한 토큰 비교
- 파싱 가능한 코드의 AST 구조 비교
- Jupyter Notebook의 코드 셀만 추출하여 비교
- 파일명보다 실제 코드 내용을 우선하여 판정
- 로컬에 더 최신 구현이 있으면 이전 Notion 버전은 중복으로 처리

## 새로 저장한 코드

| 경로 | 설명 | 작성자 |
|---|---|---|
| `src/data_collection/collect_naver_reviews_under_200.py` | 리뷰 200개 미만 매장 확인 및 리뷰 수집 | 이용현 |
| `src/data_collection/collect_blueribbon_store_names.py` | 블루리본 선정 매장명 수집 | 김동혁 |
| `src/data_collection/collect_non_blueribbon_store_names.py` | 블루리본 미선정 매장명 수집 | 김동혁 |
| `src/data_collection/recover_missing_reviews_recommended.py` | 누락 리뷰 추천순 복구 | 이용현, 조하민 |
| `src/data_collection/recover_missing_reviews_latest.py` | 누락 리뷰 최신순 복구 | 이용현, 조하민 |
| `src/feature_engineering/build_clip_interior_features.py` | CLIP 내부 이미지 선별 및 분위기 점수화 | 이용현 |
| `src/modeling/search_lightgbm_feature_subsets.py` | 상관 변수 제거 조합과 LightGBM 튜닝 | 김동혁 |
| `src/modeling/train_xgboost_regularized.py` | 과적합 억제 XGBoost 튜닝 | 이용현 |
| `src/modeling/search_xgboost_feature_subsets.py` | 상관 변수 제거 조합과 XGBoost 튜닝 | 박준수 |
| `archive/snippets/xgboost_optuna_objective.py` | 미완성 Optuna 목적함수 실험 조각 | 이용현 |
| `src/evaluation/generate_shap_consulting_report.py` | SHAP 기반 식당별 컨설팅 리포트 | 이용현 |
| `archive/snippets/count_unique_stores.py` | CSV 고유 매장 수 확인 | 이용현 |
| `archive/notion/final_integrated_workflow.md` | 실행 불가능한 의사 코드를 포함한 통합 작업 메모 | 이용현 |

## 작성자 속성 재검증

2026-07-21에 Notion 데이터베이스 뷰를 다시 조회한 결과, 기존 페이지의 `작성자` 속성은 비어 있지 않았습니다. 이전 확인 과정에서 사람 속성을 정상적으로 해석하지 못한 것이 원인이었습니다.

| Notion 사용자 ID | 확인된 이름 |
|---|---|
| `d2a3b102-c2fa-4f9a-bf8a-7f0ad5264167` | 김동혁 |
| `10ad872b-594c-81ab-9e04-00021650dd79` | 이용현 (`용현 이`로 표시) |
| `2f1d872b-594c-81c4-ac04-00026cbaf107` | 조하민 |
| `33cd872b-594c-8114-acd3-00025ed5ca8e` | 박준수 |

Notion 페이지의 소스 파일명과 로컬 코드 내용이 대응되는 파일은 이 작성자 속성을 기준으로 헤더와 파일 카탈로그를 수정했습니다. 데이터베이스에 일대일 대응 페이지가 없는 로컬 파일은 임의로 추정하지 않고 별도로 표시했습니다.

Notion에 일대일 대응 페이지가 없던 `ensemble_voting.py`, `duplicates_search.ipynb`, `review_eda.ipynb`, `review_score_eda.ipynb`는 사용자 확인에 따라 작성자를 `김동혁`으로 확정했습니다.

## 중복으로 제외한 코드

### 완전 중복

| Notion 항목 | 기존 로컬 파일 |
|---|---|
| 네이버 매장 리뷰 수집(블루리본X)_v1.0 | `src/data_collection/collect_naver_non_blueribbon_reviews.py` |
| 네이버 매장 리뷰 수집_v1.0 | `src/data_collection/collect_naver_blueribbon_reviews.py` |
| 리뷰 감정분석 가중+T-test+시각화 | `src/feature_engineering/build_weighted_review_features.py` |
| 리뷰 감정분석 코드(KoELECTRA) | `src/feature_engineering/build_koelectra_review_features.py` |

### 동일하거나 로컬에 더 최신 구현이 존재

| Notion 항목 | 판정 근거 |
|---|---|
| 네이버 리뷰 수집_v2.1 추천순·최신순 | 로컬 수집 노트북과 약 99.7% 동일 |
| 네이버 이미지 수동 크롤링 | `src/data_collection/retry_naver_interior_images.py`와 약 99.2% 동일 |
| 네이버 이미지 크롤링 | `src/data_collection/collect_naver_interior_images.py`와 약 99.9% 동일 |
| 블루리본 식당 업종별 EDA | `notebooks/analysis/cuisine_category_eda.ipynb`와 약 97.5% 동일 |
| 리뷰 수집_v2.0 추천순·최신순 | 로컬에 v2.1 구현이 존재 |
| LightGBM 기본 모델 | `src/modeling/train_lightgbm_baseline.py`에 동일 흐름과 추가 분석이 포함됨 |
| LGBM_HPT | `archive/experiments/lightgbm_bayesian_experiment.py`에 동일 목적함수와 추가 평가가 포함됨 |

## 저장하지 않은 빈 항목

- Random Forest
- XGBoost

두 Notion 항목은 코드 블록은 존재하지만 내용이 비어 있어 파일을 생성하지 않았습니다.

캐치테이블 정보 수집 코드는 프로젝트 결과에 사용하지 않은 것으로 확인되어 2026-07-20에 관련 로컬 코드와 데이터를 삭제했습니다.
