# 전체 파일 설명 및 공개 정책

현재 `code` 저장소에 있는 코드와 노트북의 역할, 작성자, 이전 파일명, GitHub 포함 여부를 정리한 문서입니다. 작성자는 2026-07-21에 Notion `ML` 데이터베이스의 `작성자` 속성과 소스 파일명을 다시 대조했습니다. Notion에 일대일 대응 항목이 없던 로컬 파일 네 개는 사용자 확인에 따라 `김동혁`으로 기록했습니다.

## 파일명이 바뀌지 않은 파일

다음 파일은 폴더 위치만 달라졌고 **파일명 자체는 유지**했습니다.

| 파일명 | 이전 경로 | 현재 경로 |
|---|---|---|
| `collect_blueribbon_store_names.py` | `notion_imports/data_collection/` | `src/data_collection/` |
| `collect_non_blueribbon_store_names.py` | `notion_imports/data_collection/` | `src/data_collection/` |
| `recover_missing_reviews_latest.py` | `notion_imports/data_collection/` | `src/data_collection/` |
| `recover_missing_reviews_recommended.py` | `notion_imports/data_collection/` | `src/data_collection/` |
| `count_unique_stores.py` | `notion_imports/utilities/` | `archive/snippets/` |
| `DATA_SOURCES.md` | 저장소 루트 | `docs/` |

다음 파일은 **파일명과 경로가 모두 유지**되었습니다.

- `.gitignore`
- `.env.example`
- `gitmessage.txt`
- `.github/ISSUE_TEMPLATE/default-issue-template.md`

`results/`에 있던 PNG 9개도 파일명은 그대로 유지하고 폴더만 `artifacts/figures/`로 변경했습니다.

## Python 코드

| 현재 파일 | 이전 파일 | 설명 | 작성자 | GitHub |
|---|---|---|---|---|
| `src/data_collection/collect_blueribbon_store_names.py` | 동일 파일명 | 블루리본 사이트에서 선정 매장명 수집 | 김동혁 | 제외 |
| `src/data_collection/collect_non_blueribbon_store_names.py` | 동일 파일명 | 블루리본 미선정 매장명 수집 | 김동혁 | 제외 |
| `src/data_collection/collect_naver_blueribbon_reviews.py` | `collect/blueribbon_review_collect.py` | 선정 매장의 네이버 방문자 리뷰 수집 | 김동혁 | 제외 |
| `src/data_collection/collect_naver_non_blueribbon_reviews.py` | `collect/no_blueribbon_review_collect.py` | 미선정 매장의 네이버 방문자 리뷰 수집 | 김동혁 | 제외 |
| `src/data_collection/collect_naver_reviews_under_200.py` | `notion_imports/data_collection/collect_reviews_under_200.py` | 리뷰 200개 미만 매장 확인 및 리뷰 수집 | 이용현 | 제외 |
| `src/data_collection/collect_naver_interior_images.py` | `collect/crawl_interior_images.py` | 네이버 플레이스 내부 이미지 수집 | 김동혁 | 제외 |
| `src/data_collection/retry_naver_interior_images.py` | `collect/crawl_interior_images_hybrid.py` | 실패한 내부 이미지 수집 재시도 | 이용현 | 제외 |
| `src/data_collection/recover_missing_reviews_latest.py` | 동일 파일명 | 누락 리뷰 최신순 복구 | 이용현, 조하민 | 제외 |
| `src/data_collection/recover_missing_reviews_recommended.py` | 동일 파일명 | 누락 리뷰 추천순 복구 | 이용현, 조하민 | 제외 |
| `src/feature_engineering/build_koelectra_review_features.py` | `review_senti_v2.0.py` | 키워드 문맥 추출 및 KoELECTRA 감성 특징 생성 | 김동혁 | 포함 |
| `src/feature_engineering/build_weighted_review_features.py` | `analyze_significance.py` | 언급 빈도와 통계 검정을 반영한 가중 특징 생성 | 김동혁 | 포함 |
| `src/feature_engineering/build_clip_interior_features.py` | `notion_imports/features/clip_interior_scoring.py` | CLIP 기반 내부 이미지 특징 생성 | 이용현 | 포함 |
| `src/modeling/train_lightgbm_baseline.py` | `ml_lgbm.py` | LightGBM 기준 분류 모델 | 김동혁 | 포함 |
| `src/modeling/train_lightgbm_optuna.py` | `ml_lgbm_optuna.py` | Optuna 기반 LightGBM 튜닝 및 평가 | 김동혁 | 포함 |
| `src/modeling/search_lightgbm_feature_subsets.py` | `notion_imports/modeling/lightgbm_correlated_feature_search.py` | 상관 특징 제거 조합과 LightGBM 탐색 | 김동혁 | 포함 |
| `src/modeling/train_xgboost_regularized.py` | `notion_imports/modeling/xgboost_regularized_optuna.py` | 정규화를 적용한 XGBoost 튜닝 | 이용현 | 포함 |
| `src/modeling/search_xgboost_feature_subsets.py` | `notion_imports/modeling/xgboost_correlated_feature_search.py` | 상관 특징 제거 조합과 XGBoost 탐색 | 박준수 | 포함 |
| `src/modeling/train_soft_voting_ensemble.py` | `ensemble_voting.py` | LightGBM·XGBoost 소프트 보팅 앙상블 | 김동혁 | 포함 |
| `src/evaluation/generate_shap_consulting_report.py` | `notion_imports/evaluation/shap_restaurant_consulting.py` | SHAP 기반 식당별 강점·약점 리포트 | 이용현 | 포함 |
| `archive/experiments/lightgbm_bayesian_experiment.py` | `bayesian_test.py` | 과거 LightGBM 베이지안 탐색 실험 | 김동혁 | 포함·보관용 |
| `archive/legacy/recover_missing_reviews.py` | `collect/fail_review_re.py` | 초기 누락 리뷰 복구 크롤러 | 이용현, 조하민 | 제외 |
| `archive/snippets/xgboost_optuna_objective.py` | `notion_imports/modeling/optuna_objective_snippet.py` | 미완성 XGBoost Optuna 목적함수 | 이용현 | 포함·보관용 |
| `archive/snippets/count_unique_stores.py` | 동일 파일명 | CSV의 고유 매장 수 확인 조각 | 이용현 | 포함·보관용 |

## Jupyter Notebook

| 현재 파일 | 이전 파일 | 설명 | 작성자 | GitHub |
|---|---|---|---|---|
| `notebooks/data_collection/collect_naver_reviews_latest.ipynb` | `collect/review_collect_최신순.ipynb` | 네이버 리뷰 최신순 수집 | 김동혁 | 제외 |
| `notebooks/data_collection/collect_naver_reviews_recommended.ipynb` | `collect/review_collect_추천순.ipynb` | 네이버 리뷰 추천순 수집 | 김동혁 | 제외 |
| `notebooks/data_collection/recover_missing_reviews_latest.ipynb` | `collect/fail_review_re_최신순.ipynb` | 누락 리뷰 최신순 복구 | 이용현, 조하민 | 제외 |
| `notebooks/data_collection/recover_missing_reviews_recommended.ipynb` | `collect/fail_review_re_추천순.ipynb` | 누락 리뷰 추천순 복구 | 이용현, 조하민 | 제외 |
| `notebooks/data_preparation/audit_duplicate_stores.ipynb` | `duplicates_search.ipynb` | 매장·리뷰 중복, 누락 및 결합 상태 점검 | 김동혁 | 포함 |
| `notebooks/data_preparation/prepare_inference_features.ipynb` | `모델테스트용_처리.ipynb` | 신규 매장의 모델 추론 특징 준비 | 이용현 | 포함 |
| `notebooks/analysis/review_text_eda.ipynb` | `review_eda.ipynb` | 리뷰 텍스트 빈도와 키워드 EDA | 김동혁 | 포함 |
| `notebooks/analysis/cuisine_category_eda.ipynb` | `review_eda_한중일_v0.1.ipynb` | 한식·중식·일식 등 업종 분류 EDA | 김동혁 | 포함 |
| `notebooks/analysis/review_feature_eda.ipynb` | `review_score_eda.ipynb` | 가중 리뷰 특징 분포 및 집단 비교 | 김동혁 | 포함 |
| `notebooks/modeling/lightgbm_optuna_experiments.ipynb` | `ml_lgbm_optuna.ipynb` | 특징 조합, Optuna, 임계값 모델링 기록 | 김동혁 | 포함 |

## 문서와 설정

| 파일 | 설명 | GitHub |
|---|---|---|
| `README.md` | 프로젝트 개요, 구조, 실행 순서 | 포함 |
| `requirements.txt` | Python 패키지 목록 | 포함 |
| `.env.example` | 로컬 경로 환경변수 예시 | 포함 |
| `.gitignore` | 데이터, 수집 코드, 비밀정보, 생성물 제외 정책 | 포함 |
| `docs/DATA_SOURCES.md` | Google Drive와 로컬 데이터 계보 | 포함 |
| `docs/FILE_MIGRATION.md` | 대표적인 기존 경로와 새 경로 대응표 | 포함 |
| `docs/FILE_CATALOG.md` | 전체 파일의 역할, 작성자, 이전 이름, 공개 여부 | 포함 |
| `docs/NOTION_CODE_AUDIT.md` | Notion 코드와 로컬 코드 비교 기록 | 포함 |
| `archive/notion/final_integrated_workflow.md` | Notion에서 복원한 통합 작업 메모와 의사 코드 | 제외 |
| `gitmessage.txt` | 팀 커밋 메시지 작성 템플릿 | 포함 |
| `.github/ISSUE_TEMPLATE/default-issue-template.md` | 기존 GitHub Issue 템플릿 | 포함 |

## 결과 이미지

`artifacts/figures/`의 PNG만 검토된 공개 결과물로 허용합니다.

- `blue_ribbon_count_comparison_overall.png`: 전체 집단의 키워드 언급 수 비교
- `blue_ribbon_count_comparison_category.png`: 업종별 키워드 언급 수 비교
- `blue_ribbon_ratio_comparison_overall.png`: 전체 집단의 키워드 언급 비율 비교
- `blue_ribbon_ratio_comparison_category.png`: 업종별 키워드 언급 비율 비교
- `blue_ribbon_score_comparison_overall.png`: 전체 집단의 감성 점수 비교
- `blue_ribbon_score_comparison_category.png`: 업종별 감성 점수 비교
- `blue_ribbon_weighted_score_comparison_overall.png`: 전체 집단의 가중 점수 비교
- `blue_ribbon_weighted_score_comparison_category.png`: 업종별 가중 점수 비교
- `lgbm_tree_structure_index0.png`: LightGBM 첫 번째 트리 구조

## GitHub 제외 대상

다음 항목은 로컬에는 남아 있지만 GitHub에는 올라가지 않습니다.

- 모든 CSV, Excel, Parquet, NumPy 데이터 파일
- 원본 이미지와 압축 파일
- 학습 모델과 체크포인트
- `.env` 및 로컬 IDE·Jupyter 임시파일
- `src/data_collection/`
- `notebooks/data_collection/`
- `archive/legacy/`
- `archive/notion/`
- 크롤링 완료·실패 목록과 로그
- 노트북 실행 결과의 로컬 백업(`outputs/notebook_backups/`)

공개 노트북은 코드와 Markdown 설명만 포함하며, 리뷰 원문·데이터 미리보기·모델 실행 로그·내장 이미지는 출력 셀에서 제거했습니다.

## 사용하지 않아 삭제한 파일

2026-07-20에 프로젝트 결과에 사용하지 않은 것으로 확인된 다음 캐치테이블 전용 파일을 로컬에서도 삭제했습니다.

- `code/catchtable/` 전체
- `notebooks/data_collection/collect_catchtable_store_info.ipynb`
- `data/sample/blueribbon_catchtable_강북일부.csv`

리뷰에 자연어로 포함된 “캐치테이블” 표현과 대기·예약 키워드 사전은 모델 특징 생성에 사용될 수 있으므로 유지했습니다.
