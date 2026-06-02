import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
import pandas as pd
import time
import re
import os

# =====================================================================
# 설정값
# =====================================================================
MAX_REVIEWS_PER_STORE = 200  
RESULT_FILE = "naver_place_reviews_detailed.csv"
COMPLETED_FILE = "completed_stores.txt" # 한 번 처리한 매장을 기억할 파일

def human_sleep(sec=2):
    time.sleep(sec)

def extract_dong(address):
    match = re.search(r'\((.*?)\)', str(address))
    if match:
        return match.group(1).strip()
    return ""

# [신규] 완료된 매장 기록을 불러오는 함수
def load_completed_stores():
    if os.path.exists(COMPLETED_FILE):
        with open(COMPLETED_FILE, 'r', encoding='utf-8') as f:
            return set(line.strip() for line in f if line.strip())
    return set()

# [신규] 완료된 매장을 기록하는 함수
def save_completed_store(store_name):
    with open(COMPLETED_FILE, 'a', encoding='utf-8') as f:
        f.write(store_name + "\n")

# [신규] 기존 CSV 파일이 있으면 불러오는 함수
def load_existing_results():
    if os.path.exists(RESULT_FILE):
        return pd.read_csv(RESULT_FILE).to_dict('records')
    return []

def get_naver_place_reviews(df):
    # 1. 기존 진행 상황 불러오기
    completed_stores = load_completed_stores()
    final_data = load_existing_results()
    
    print(f"\n[복구 시스템] 이전에 완료한 매장 {len(completed_stores)}개를 건너뜁니다.")
    print(f"[복구 시스템] 기존에 수집된 리뷰 {len(final_data)}개를 불러왔습니다.")
    
    options = uc.ChromeOptions()
    options.add_argument('--window-size=1920,1080')
    # 메모리 부족 방지 옵션 추가
    options.add_argument('--disable-dev-shm-usage') 
    
    driver = uc.Chrome(options=options, use_subprocess=True, version_main=148)
    
    for index, row in df.iterrows():
        store_name = str(row['매장명']).strip()
        store_address = str(row['주소']).strip()
        
        # ⭐ 이미 처리한 매장이라면 1초의 망설임도 없이 패스!
        if store_name in completed_stores:
            continue
            
        dong = extract_dong(store_address)
        search_query = f"{dong} {store_name}".strip() if dong else store_name
        
        print(f"\n[{index+1}/{len(df)}] '{search_query}' 탐색 중...")
        
        # ⭐ [핵심 방어] 매번 네이버 지도를 새로 접속하여 메모리 찌꺼기 완벽 제거
        driver.get("https://map.naver.com/v5/")
        time.sleep(4)
        
        # --- 1. 검색어 입력 ---
        driver.switch_to.default_content()
        try:
            search_box = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "input.input_search"))
            )
            # 새로고침 했으므로 지우는 과정 없이 바로 입력
            search_box.send_keys(search_query)
            search_box.send_keys(Keys.ENTER)
            time.sleep(3) 
        except Exception:
            print(f"    ⚠️ 검색창을 찾지 못했습니다. (일시적 오류일 수 있으므로 다음 매장으로 넘어갑니다)")
            # 에러가 났을 때는 completed_stores에 넣지 않고 스킵 (다음 번에 다시 시도하도록)
            continue

        # --- 2. 검색 결과 프레임 확인 ---
        driver.switch_to.default_content()
        success = False
        try:
            driver.switch_to.frame("searchIframe")
            store_list = driver.find_elements(By.XPATH, "//li[.//a]")
            if len(store_list) > 0:
                print("    ✔️ 검색 목록 발견! 첫 번째 매장 클릭.")
                driver.execute_script("arguments[0].click();", store_list[0].find_element(By.XPATH, ".//a"))
                time.sleep(3)
                driver.switch_to.default_content()
                driver.switch_to.frame("entryIframe")
                success = True
        except Exception:
            driver.switch_to.default_content()
            try:
                driver.switch_to.frame("entryIframe")
                driver.find_element(By.XPATH, "//*[contains(text(), '방문자')]")
                success = True
            except:
                pass

        if not success:
            print("    ❌ 이 매장은 존재하지 않습니다.")
            save_completed_store(store_name) # 없는 매장도 '완료' 처리하여 다신 안 찾음
            continue
            
        # --- 3. 방문자 리뷰 수 확인 ---
        try:
            review_count_element = WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.XPATH, "//*[contains(text(), '방문자 리뷰') or contains(text(), '방문자리뷰')]"))
            )
            numbers = re.findall(r'\d+', review_count_element.text.replace(',', ''))
            review_count = int(numbers[0]) if numbers else 0
            if review_count < 100:
                print(f"    ❌ 리뷰 수 부족 ({review_count}개). 건너뜁니다.")
                save_completed_store(store_name) # 리뷰 부족도 '완료' 처리
                continue
        except:
            save_completed_store(store_name)
            continue
            
        # --- 4. 리뷰 탭 클릭 ---
        try:
            review_tab = WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.XPATH, "//a[@role='tab']//span[text()='리뷰']"))
            )
            driver.execute_script("arguments[0].click();", review_tab)
            human_sleep(2)
        except:
            save_completed_store(store_name)
            continue
            
        # --- 5. 최신순 정렬 ---
        try:
            latest_btn = WebDriverWait(driver, 4).until(
                EC.presence_of_element_located((By.XPATH, "//*[text()='최신순']"))
            )
            driver.execute_script("arguments[0].click();", latest_btn)
            human_sleep(1.5)
        except:
            pass

        # --- 6. 실시간 스캔 수집 ---
        print(f"    🔄 리뷰 수집 시작 (목표: {MAX_REVIEWS_PER_STORE}개)")
        collected_reviews = 0
        consecutive_empty_scrolls = 0  
        scraped_signatures = set() 
        
        while collected_reviews < MAX_REVIEWS_PER_STORE:
            try:
                text_more_btns = driver.find_elements(By.XPATH, "//*[text()='내용 더보기']")
                for btn in text_more_btns:
                    driver.execute_script("arguments[0].click();", btn)
            except:
                pass
                
            review_items = driver.find_elements(By.XPATH, "//li[.//time]")
            new_this_round = 0
            
            for item in review_items:
                if collected_reviews >= MAX_REVIEWS_PER_STORE:
                    break
                    
                try:
                    item_text = driver.execute_script("return arguments[0].innerText;", item)
                    lines = [ln.strip() for ln in item_text.split('\n') if ln.strip()]
                    if not lines: continue
                    
                    rev_userid = lines[0]
                    if "리뷰" in rev_userid or "추천순" in rev_userid:
                        rev_userid = lines[1] if len(lines) > 1 else ""
                        
                    date_nodes = item.find_elements(By.XPATH, ".//time")
                    rev_date = driver.execute_script("return arguments[0].innerText;", date_nodes[0]).strip() if date_nodes else ""
                    
                    star_nodes = item.find_elements(By.XPATH, ".//span[text()='별점']/..")
                    rev_rating = driver.execute_script("return arguments[0].innerText;", star_nodes[0]).replace("별점", "").replace("\n", "").strip() if star_nodes else ""
                    
                    rev_content = ""
                    content_nodes = item.find_elements(By.XPATH, ".//a[@data-pui-click-code='rvshowmore'] | .//span[contains(@class, 'zPfVt')]")
                    
                    if content_nodes:
                        longest_text = ""
                        for node in content_nodes:
                            node_text = driver.execute_script("return arguments[0].innerText;", node).strip()
                            if len(node_text) > len(longest_text):
                                longest_text = node_text
                        rev_content = longest_text
                        
                        rev_content = rev_content.replace("내용 더보기", "").replace("펼쳐서 더보기", "").replace("접기", "").strip()
                        rev_content = rev_content.replace("\n", " ")
                        rev_content = re.sub(r'\s+', ' ', rev_content)
                        
                    if not rev_content:
                        rev_content = "텍스트 리뷰 없음"
                        
                    signature = f"{rev_userid}_{rev_content[:15]}"
                    if signature in scraped_signatures:
                        continue 
                        
                    scraped_signatures.add(signature)
                    
                    final_data.append({
                        '매장명': store_name,
                        '유저ID': rev_userid,
                        '날짜': rev_date,
                        '별점': rev_rating,
                        '리뷰 내용': rev_content
                    })
                    collected_reviews += 1
                    new_this_round += 1
                    
                except Exception:
                    continue
                    
            if collected_reviews < MAX_REVIEWS_PER_STORE:
                try:
                    more_btn = driver.find_element(By.XPATH, "//*[text()='펼쳐서 더보기']")
                    driver.execute_script("arguments[0].scrollIntoView(true);", more_btn)
                    driver.execute_script("window.scrollBy(0, -100);") 
                    time.sleep(0.5)
                    driver.execute_script("arguments[0].click();", more_btn)
                    time.sleep(2.0) 
                    consecutive_empty_scrolls = 0
                except:
                    driver.execute_script("window.scrollBy(0, -500);")
                    time.sleep(0.5)
                    driver.execute_script("window.scrollBy(0, 1500);")
                    time.sleep(2.0)
                    
                    if new_this_round == 0:
                        consecutive_empty_scrolls += 1
                        if consecutive_empty_scrolls > 5: 
                            break
                    else:
                        consecutive_empty_scrolls = 0
                        
        print(f"    ✅ '{store_name}' 상세 데이터 {collected_reviews}개 성공적으로 수집 완료!")
        
        # ⭐ [실시간 저장] 매장 하나가 끝날 때마다 진행 상황을 즉시 저장합니다.
        pd.DataFrame(final_data, columns=['매장명', '유저ID', '날짜', '별점', '리뷰 내용']).to_csv(RESULT_FILE, index=False, encoding="utf-8-sig")
        save_completed_store(store_name) # 다 끝났으니 완료 목록에 도장 쾅!

    driver.quit()
    print("\n🎉 모든 수집이 완료되었습니다!")
    return pd.DataFrame(final_data, columns=['매장명', '유저ID', '날짜', '별점', '리뷰 내용'])

if __name__ == "__main__":
    exclude_list = ['베이커리', '카페', '커피전문점', '에스프레소바', '전통차전문점', '홍차전문점', '중국차전문점', '일본차전문점', '티카페', '북카페', '과일카페', '주스전문점', '체험카페', '디저트전문점', '라이브디저트', '한식디저트', '일본디저트', '중국디저트', '대만디저트', '구움과자', '그래놀라', '꽈배기', '단팥죽', '도넛', '떡', '떡카페', '떡케이크', '마카롱', '바움쿠헨', '빙수', '사탕', '센베이', '스콘', '아사이볼', '아이스크림', '양갱', '에그타르트', '요거트', '잼', '젤라토', '찐빵', '찹쌀떡', '초콜릿', '츄러스', '카스텔라', '카이막', '캐러멜', '컵케이크', '케이크', '쿠키', '타르트', '티라미수', '파이', '팥빵', '푸딩', '한과', '호두과자', '호떡', '후르츠산도', '베이커리', '페이스트리', '고로케', '미트파이', '베이글', '바스코티', '식빵', '와플', '샌드위치', '소금빵', '크레프', '크로플', '크루아상', '팬케이크']
    df = pd.read_csv("blueribbon_data_강북일부.csv")
    df = df[~df['카테고리'].isin(exclude_list)]
    
    print("\n[시작] 네이버 플레이스 방문자 리뷰 맞춤형 크롤링...")
    review_df = get_naver_place_reviews(df)