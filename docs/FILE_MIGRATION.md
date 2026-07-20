# 파일 구조 변경 기록

GitHub 공개를 위해 2026-07-20에 수행한 주요 파일 이동 및 이름 변경 내역입니다. 데이터 파일은 이동하거나 이름을 변경하지 않았습니다.

| 기존 경로 | 새 경로 |
|---|---|
| `collect/blueribbon_review_collect.py` | `src/data_collection/collect_naver_blueribbon_reviews.py` |
| `collect/no_blueribbon_review_collect.py` | `src/data_collection/collect_naver_non_blueribbon_reviews.py` |
| `collect/crawl_interior_images.py` | `src/data_collection/collect_naver_interior_images.py` |
| `collect/crawl_interior_images_hybrid.py` | `src/data_collection/retry_naver_interior_images.py` |
| `review_senti_v2.0.py` | `src/feature_engineering/build_koelectra_review_features.py` |
| `analyze_significance.py` | `src/feature_engineering/build_weighted_review_features.py` |
| `ml_lgbm.py` | `src/modeling/train_lightgbm_baseline.py` |
| `ml_lgbm_optuna.py` | `src/modeling/train_lightgbm_optuna.py` |
| `ensemble_voting.py` | `src/modeling/train_soft_voting_ensemble.py` |
| `duplicates_search.ipynb` | `notebooks/data_preparation/audit_duplicate_stores.ipynb` |
| `모델테스트용_처리.ipynb` | `notebooks/data_preparation/prepare_inference_features.ipynb` |
| `review_eda.ipynb` | `notebooks/analysis/review_text_eda.ipynb` |
| `review_eda_한중일_v0.1.ipynb` | `notebooks/analysis/cuisine_category_eda.ipynb` |
| `review_score_eda.ipynb` | `notebooks/analysis/review_feature_eda.ipynb` |
| `ml_lgbm_optuna.ipynb` | `notebooks/modeling/lightgbm_optuna_experiments.ipynb` |

Notion에서 복원한 실행 가능한 코드는 역할에 따라 `src/` 아래로 합쳤습니다. 미완성 코드 조각과 통합 작업 메모는 출처 정보를 유지한 채 `archive/`로 옮겼습니다.

## 삭제한 미사용 파일

캐치테이블 관련 코드와 데이터는 최종 프로젝트에서 사용하지 않은 것으로 확인되어 2026-07-20에 삭제했습니다.

- 기존 `catchtable/` 폴더 전체
- `notebooks/data_collection/collect_catchtable_store_info.ipynb`
- `../data/sample/blueribbon_catchtable_강북일부.csv`
