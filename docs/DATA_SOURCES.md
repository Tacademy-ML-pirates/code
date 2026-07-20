# 데이터 소스 및 저장 정책

## 원본 위치

- Google Drive 폴더: [T아카데미_ML_blueribbon](https://drive.google.com/drive/folders/12DjeIDFp103VEpUEExMaS7mESkHR-A4b)
- 로컬 데이터 루트: `../data`
- 코드 저장소에는 원본 데이터와 대용량 모델 산출물을 커밋하지 않습니다.

Google Drive를 데이터 원본 저장소로 사용하고, 로컬 데이터는 분석과 실행을 위한 작업 사본으로 취급합니다.

## Drive 구조 요약

```text
T아카데미_ML_blueribbon/
├── model datasets and derived features
├── review/
│   ├── 한식/    # 6 files
│   ├── 중식/    # 6 files
│   ├── 일식/    # 6 files
│   └── 양식/    # 6 files
├── store_info/  # 8 files
├── yes_blue_한식/   # 137 restaurant folders
├── yes_blue_일식/   # 137 restaurant folders
├── no_blue_한식/    # 113 restaurant folders
├── no_blue_일식/    # 117 restaurant folders
├── no_blue_중식/    # 124 restaurant folders
└── no_blue_양식/    # 185 restaurant folders
```

`yes_blue_중식`과 `yes_blue_양식` 아래의 추가 이미지 폴더는 현재 비어 있습니다. `new식당v2_리뷰모음(최신&추천)` 폴더도 현재 비어 있습니다.

## 핵심 데이터 매핑

| 역할 | Drive 파일 | 로컬 위치 | GitHub 정책 |
|---|---|---|---|
| 모델 데이터 v1.1 | `블루리본_최최종_마참내_v1.1.csv` | `../data/블루리본_최최종_마참내_v1.1.csv` | 제외 |
| 모델 데이터 v1.1 중복 제거 | `블루리본_최최종_마참내_v1.1.1_중복매장제거.csv` | `../data/블루리본_최최종_마참내_v1.1.1.csv` | 제외 |
| 모델 데이터 v1.2 | `블루리본_최최종_마참내_v1.2.csv` | `../data/블루리본_최최종_마참내_v1.2.csv` | 제외 |
| 모델 데이터 v1.2 중복 제거 | `블루리본_최최종_마참내_v1.2.1_중복매장제거.csv` | `../data/블루리본_최최종_마참내_v1.2.1.csv` | 제외 |
| 최종 가중 리뷰 특징 | `KoELECTRA_X_feature_v2.3_weighted.csv` | `../data/KoELECTRA_X_feature_v2.3_weighted.csv` | 제외 |
| 전체 KoELECTRA 결과 | `KoELECTRA_cpu_score.csv` (약 224 MB) | `../data/KoELECTRA_cpu_score.csv` | 반드시 제외 |
| 통합 원본 | `merge_blue.xlsx` (약 40.8 MB) | `../data/merge_blue.xlsx` | 제외 |
| 전체 감성분석 결과 | `전체리뷰_감정분석.csv` (약 86.8 MB) | `../리뷰_감정분석.csv` | 반드시 제외 |
| 식당별 감성 요약 | `매장별_리뷰_감성분석_summary.csv` | `../매장별_감정분석_summary.csv` | 필요 시 샘플만 공개 |
| T-test 결과 | `T_test_results_detailed.csv` | `../data/T_test_results_detailed.csv` | 필요 시 결과물로 공개 가능 |
| 테스트 특징 | `KoELECTRA_X_feature_new_test_weighted(noblue).csv` | `../data/model_test_data/` | 필요 시 비식별 샘플만 공개 |

Drive와 로컬에서 파일명이 조금 다른 중복 제거 데이터 및 감성 요약 파일은 파일 크기가 일치합니다. 이후 정리 단계에서 의미가 드러나는 표준 파일명으로 변경하되, 데이터 계보 문서에 기존 이름을 함께 기록합니다.

## 이미지 데이터

Drive에는 음식 유형 및 블루리본 여부별로 수백 개 매장 이미지 폴더가 있습니다. 로컬에는 일부 작업 사본만 있으므로 Drive를 이미지 원본으로 유지합니다.

- 로컬 블루리본 이미지 폴더: 137개 매장
- 로컬 비블루리본 이미지 폴더: 98개 매장
- Drive에서 확인된 블루리본 이미지 폴더: 274개 매장
- Drive에서 확인된 비블루리본 이미지 폴더: 539개 매장

이미지는 용량과 출처 문제로 GitHub에 업로드하지 않습니다. README에는 데이터 수집 방법, 이미지 개수, 전처리 방법과 소량의 허가된 예시만 포함합니다.

## 코드에서 사용할 환경변수

로컬에서는 `.env.example`을 `.env`로 복사한 뒤 실제 경로를 설정합니다. `.env`는 Git에서 제외됩니다.

```text
DATA_DIR
REVIEW_DATA_DIR
STORE_INFO_DIR
BLUE_IMAGE_DIR
NON_BLUE_IMAGE_DIR
MODEL_DIR
OUTPUT_DIR
DRIVE_ROOT
```

Colab에서 Drive 폴더를 사용할 경우 공유 폴더를 `내 드라이브`에 바로가기 추가한 뒤 `DRIVE_ROOT`를 실제 마운트 경로에 맞게 수정합니다.

## 기준 데이터셋 검증 결과

2026-07-20에 로컬 파일과 발표·모델 코드를 다시 대조했습니다.

| 구분 | v1.2 | v1.2.1 |
|---|---:|---:|
| 행 수 | 1,440 | 1,428 |
| 열 수 | 22 | 22 |
| 고유 매장 수 | 1,428 | 1,428 |
| 중복 매장명 행 수 | 12 | 0 |
| 블루리본 미선정/선정 | 696 / 744 | 689 / 739 |

v1.2.1은 v1.2와 컬럼 구성이 같고, 중복 매장 12건을 제거한 정제본입니다. Drive의 원래 파일명도 `블루리본_최최종_마참내_v1.2.1_중복매장제거.csv`로 확인됩니다.

- **발표 결과 재현 기준:** `v1.2` — 발표자료의 전체 1,440건, 학습 1,152건, 테스트 288건과 일치하며 최종 LightGBM 및 앙상블 코드가 이 파일을 참조합니다.
- **향후 재학습 권장 기준:** `v1.2.1` — 매장 단위 중복을 제거해 동일 매장이 학습·테스트에 나뉘는 데이터 누수 위험을 줄인 최신 정제본입니다.

따라서 기존 발표 지표를 재현할 때는 v1.2를 유지하고, 모델을 새로 학습하거나 성능을 다시 보고할 때는 v1.2.1을 사용해 지표를 재산출합니다. 두 데이터셋의 결과를 같은 실험으로 혼용하지 않습니다.
