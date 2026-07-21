# Blue Ribbon Restaurant Prediction

소비자 리뷰와 식당 내부 이미지에서 추출한 특징을 이용해 블루리본 선정 가능성을 예측하고, 변수 중요도와 SHAP 값으로 식당별 강점과 개선 지점을 제시하는 머신러닝 프로젝트입니다.

## 프로젝트 목표

- 리뷰의 가족 모임, 기념일, 맛, 서비스, 분위기, 가격 등 주요 키워드별 긍정 반응을 점수화합니다.
- 식당 내부 이미지에서 고급성, 쾌적성, 감성 특징을 추출합니다.
- LightGBM과 XGBoost 기반 분류 모델로 블루리본 선정 여부를 예측합니다.
- 변수 중요도와 SHAP 분석을 활용해 식당별 컨설팅 근거를 생성합니다.

## 저장소 구조

```text
.
├── src/
│   ├── data_collection/       # 로컬 전용 수집 코드(Git 제외)
│   ├── feature_engineering/   # KoELECTRA, CLIP, 가중 리뷰 특징 생성
│   ├── modeling/              # LightGBM, XGBoost, 앙상블 학습
│   └── evaluation/            # SHAP 기반 예측 해석 및 컨설팅
├── notebooks/
│   ├── data_collection/       # 수집 실험 노트북
│   ├── data_preparation/      # 중복 점검 및 추론 데이터 준비
│   ├── analysis/              # 리뷰와 특징 EDA
│   └── modeling/              # 모델 탐색 과정과 실행 결과
├── artifacts/figures/         # 공개 가능한 분석 및 모델 그림
├── docs/                      # 데이터 출처와 코드 이관 기록
└── archive/                   # 미완성 조각, 과거 실험, Notion 작업 메모
```

`archive/`의 파일은 작업 이력 보존용이며 현재 실행 파이프라인에 포함되지 않습니다.
`src/data_collection/`, `notebooks/data_collection/`, `archive/legacy/`, `archive/notion/`은 로컬에만 보관하며 `.gitignore`로 GitHub 업로드 대상에서 제외합니다.

## 주요 코드

| 단계 | 파일 | 역할 |
|---|---|---|
| 리뷰 특징 | `src/feature_engineering/build_koelectra_review_features.py` | 키워드 문맥 감성 점수 생성 |
| 특징 가중 | `src/feature_engineering/build_weighted_review_features.py` | 리뷰 수와 유의성 기반 가중 특징 생성 |
| 이미지 특징 | `src/feature_engineering/build_clip_interior_features.py` | CLIP 기반 내부 이미지 특징 생성 |
| 기준 모델 | `src/modeling/train_lightgbm_baseline.py` | LightGBM 기준 모델 학습 |
| 최종 탐색 | `src/modeling/train_lightgbm_optuna.py` | Optuna 기반 LightGBM 튜닝 |
| 앙상블 | `src/modeling/train_soft_voting_ensemble.py` | LightGBM·XGBoost 소프트 보팅 |
| 컨설팅 | `src/evaluation/generate_shap_consulting_report.py` | 매장별 SHAP 강점·약점 리포트 |

## 실행 환경

Python 3.10 이상을 권장합니다.

```bash
python -m venv .venv
pip install -r requirements.txt
```

`.env.example`을 `.env`로 복사하고 로컬 데이터 경로를 설정합니다. 현재 스크립트의 상대경로는 저장소 루트에서 실행하는 것을 기준으로 합니다.

## 데이터 준비

원본 리뷰, 이미지, 모델 학습 CSV는 용량과 데이터 출처 문제로 GitHub에 포함하지 않습니다. 기본 로컬 배치는 이 저장소와 `data` 폴더가 같은 상위 디렉터리에 있는 형태입니다.

공개 노트북은 리뷰 원문과 데이터 표가 실행 결과에 포함되지 않도록 출력 셀을 제거한 상태로 관리합니다. 로컬 실행 결과 백업은 Git에서 제외되는 `outputs/notebook_backups/`에 보관합니다.

```text
project/
├── code/    # 이 저장소
└── data/    # Git에서 제외되는 로컬 데이터
```

발표 결과는 1,440행의 `v1.2` 데이터로 생성했습니다. 향후 재학습에는 중복 매장을 제거한 1,428행의 `v1.2.1` 사용을 권장합니다. 자세한 계보와 Google Drive 매핑은 [데이터 소스 문서](docs/DATA_SOURCES.md)를 참고하세요.

## 권장 실행 순서

1. `notebooks/data_preparation/audit_duplicate_stores.ipynb`로 매장 중복과 결합 상태를 확인합니다.
2. 리뷰 특징, 이미지 특징, 가중 특징 생성 코드를 실행합니다.
3. `src/modeling/train_lightgbm_optuna.py` 또는 특징 조합 탐색 코드를 실행합니다.
4. 최종 모델을 준비한 뒤 SHAP 컨설팅 코드를 실행합니다.

수집 코드는 외부 웹사이트의 화면 구조 변경에 영향을 받을 수 있습니다. 실행 전 각 서비스의 이용약관과 로봇 정책을 확인하고 요청 간격을 유지하세요.

## 문서

- [데이터 소스 및 저장 정책](docs/DATA_SOURCES.md)
- [Notion 코드 비교 및 이관 기록](docs/NOTION_CODE_AUDIT.md)
- [기존 파일명과 새 경로 대응표](docs/FILE_MIGRATION.md)
- [전체 파일 설명·작성자·공개 여부](docs/FILE_CATALOG.md)
