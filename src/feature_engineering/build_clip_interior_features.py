"""
Description:
    CLIP으로 식당 내부 이미지를 선별하고 고급성, 쾌적성, 감성 점수를 집계합니다.

Author:
    이용현

Source:
    Notion ML database, page 37bbf34c-c2ef-800d-b7fe-eabf612e6810
"""

import os
import glob
import pandas as pd
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from tqdm import tqdm

# 구글 드라이브 기본 경로 설정
ROOT_DIR = "/content/drive/MyDrive/Colab Notebooks"

# 연산 장치(GPU/CPU) 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# CLIP 모델 및 프로세서 로드
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Stage 1: 이미지 분류 및 필터링용 프롬프트 정의
filter_prompts = [
    "A photo of a restaurant interior",
    "A close-up photo of food on a plate",
    "A photo of a restaurant menu or sign",
    "A photo of a restaurant exterior or building facade"
]

# Stage 2: 분위기 속성 세부 채점용 프롬프트 정의
scoring_prompts = [
    "A photo of a sophisticated and high-end restaurant interior with premium materials, elegant design, and refined atmosphere.",
    "A photo of a spacious and comfortable restaurant interior with widely spaced seating, neat layout, and a quiet, peaceful environment.",
    "A photo of a beautiful and aesthetic restaurant interior with moody lighting, tasteful decorations, and an impressive vibe."
]

# 최종 결과를 저장할 리스트
master_results = []

# 루트 디렉토리의 하위 폴더 순회
for cat_folder in os.listdir(ROOT_DIR):
    cat_path = os.path.join(ROOT_DIR, cat_folder)
    if not os.path.isdir(cat_path):
        continue

    # 폴더명을 기반으로 블루리본 여부 판별
    if ("yes" in cat_folder or "blue" in cat_folder) and "no" not in cat_folder:
        is_blue = "yes"
    elif "리본" in cat_folder and "노" not in cat_folder:
        is_blue = "yes"
    else:
        is_blue = "no"

    # 폴더명을 기반으로 음식 종류 판별
    cuisine = "기타"
    for c in ["양식", "중식", "일식", "일본", "한식"]:
        if c in cat_folder:
            cuisine = "일식" if c == "일본" else c
            break

    # 정의되지 않은 카테고리 폴더는 제외
    if cuisine == "기타" and not ("blue" in cat_folder or "리본" in cat_folder):
        continue

    # 카테고리 폴더 내부에 존재하는 식당 폴더 목록 가져오기
    restaurant_folders = [r for r in os.listdir(cat_path) if os.path.isdir(os.path.join(cat_path, r))]

    print(f"Processing: {cat_folder}")

    # 식당별 폴더 순회 진행
    for rest_folder in tqdm(restaurant_folders):
        rest_path = os.path.join(cat_path, rest_folder)

        # 지원하는 이미지 확장자 파일 검색
        image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(rest_path, ext)))

        total_img_count = len(image_paths)

        # 사진이 없는 식당은 결측치(NaN)로 처리 후 기록
        if total_img_count == 0:
            master_results.append({
                "식당이름": rest_folder,
                "블루리본여부": is_blue,
                "음식종류": cuisine,
                "총_사진_개수": 0,
                "통과된_내부사진_개수": 0,
                "고급성_평균": np.nan, "고급성_최대": np.nan,
                "쾌적성_평균": np.nan, "쾌적성_최대": np.nan,
                "감성_평균": np.nan, "감성_최대": np.nan
            })
            continue

        # 개별 이미지 점수를 저장할 리스트 및 카운터 초기화
        scores_luxury = []
        scores_comfort = []
        scores_mood = []
        passed_img_count = 0

        # 식당 폴더 내의 이미지 파일 순회 분석
        for img_path in image_paths:
            try:
                # 이미지 로드 후 RGB 변환
                img = Image.open(img_path).convert("RGB")

                # Stage 1: 이미지 유형 분류 진행
                inputs_filter = processor(text=filter_prompts, images=img, return_tensors="pt", padding=True).to(device)
                with torch.no_grad():
                    outputs_filter = model(**inputs_filter)
                    probs = outputs_filter.logits_per_image.softmax(dim=-1).cpu().numpy()[0]

                # 가장 높은 확률을 가진 항목이 0번(식당 내부)인 경우에만 통과
                if np.argmax(probs) == 0:
                    passed_img_count += 1

                    # Stage 2: 분위기 속성 채점 진행
                    inputs_score = processor(text=scoring_prompts, images=img, return_tensors="pt", padding=True).to(device)
                    with torch.no_grad():
                        outputs_score = model(**inputs_score)
                        img_feats = outputs_score.image_embeds
                        txt_feats = outputs_score.text_embeds

                        # 임베딩 벡터 정규화
                        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
                        txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)

                        # 코사인 유사도 연산
                        similarities = (img_feats @ txt_feats.T).squeeze(0).cpu().numpy()

                    # 속성별 점수를 임시 리스트에 기록
                    scores_luxury.append(similarities[0])
                    scores_comfort.append(similarities[1])
                    scores_mood.append(similarities[2])

            except Exception as e:
                # 파일 손상 등의 에러 발생 시 해당 이미지는 건너뜀
                continue

        # 통과된 내부 사진이 존재하면 평균 및 최대값 계산, 없으면 결측치 처리
        # (기존 반복문 하단의 통계 계산 부분)
        avg_luxury = np.mean(scores_luxury) if scores_luxury else np.nan
        med_luxury = np.median(scores_luxury) if scores_luxury else np.nan  # 중앙값 추가
        max_luxury = np.max(scores_luxury) if scores_luxury else np.nan

        avg_comfort = np.mean(scores_comfort) if scores_comfort else np.nan
        med_comfort = np.median(scores_comfort) if scores_comfort else np.nan  # 중앙값 추가
        max_comfort = np.max(scores_comfort) if scores_comfort else np.nan

        avg_mood = np.mean(scores_mood) if scores_mood else np.nan
        med_mood = np.median(scores_mood) if scores_mood else np.nan  # 중앙값 추가
        max_mood = np.max(scores_mood) if scores_mood else np.nan

        master_results.append({
            "식당이름": rest_folder,
            "블루리본여부": is_blue,
            "음식종류": cuisine,
            "총_사진_개수": total_img_count,
            "통과된_내부사진_개수": passed_img_count,
            "고급성_평균": avg_luxury, "고급성_중앙값": med_luxury, "고급성_최대": max_luxury,
            "쾌적성_평균": avg_comfort, "쾌적성_중앙값": med_comfort, "쾌적성_최대": max_comfort,
            "감성_평균": avg_mood, "감성_중앙값": med_mood, "감성_최대": max_mood
        })

# 데이터프레임 변환 후 구글 드라이브에 CSV 파일로 저장
df_final = pd.DataFrame(master_results)
output_csv_path = os.path.join(ROOT_DIR, "블루리본_추가_사진분위기_dataset.csv")
df_final.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
print(f"Saved: {output_csv_path}")
