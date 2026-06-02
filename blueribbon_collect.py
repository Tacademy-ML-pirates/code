from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import pandas as pd
import time


def get_all_blue_ribbon_data():
    # 1. 크롬 드라이버 설정 (봇 탐지 우회)
    options = webdriver.ChromeOptions()
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
        "source": """ Object.defineProperty(navigator, 'webdriver', { get: () => undefined }) """
    })
    
    # 2. URL 접속
    url = "https://bluer.co.kr/"
    driver.get(url)
    
    # =====================================================================
    # 3. 사용자 수동 개입 대기
    # =====================================================================
    print("\n[안내] 브라우저가 열렸습니다. 필요하다면 로그인을 진행해 주세요.")
    input("준비가 완료되면 이 창에서 [Enter] 키를 누르세요...")
    print("\n본격적인 크롤링을 시작합니다! (블루리본이 끝날 때까지 진행됩니다.)")
    
    scraped_data = [] 
    current_page = 1
    
    # 4. 크롤링 로직 (무한 반복)
    while True:
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        items = soup.select('div.thumb-restaurant')
        
        # 현재 페이지에서 발견된 블루리본 매장 수를 세기 위한 변수
        blue_ribbon_count_on_page = 0
        
        for item in items:
            ribbon_list = item.select('ul.ribbons li')
            
            # 블루리본이 있는 경우에만 수집
            if len(ribbon_list) > 0:
                blue_ribbon_count_on_page += 1
                
                name_row = item.select_one('div.header-name-row')
                address_row = item.select_one('div.content-info.juso-info')
                
                # [추가됨] 카테고리 추출 (ol.foodtype 안의 첫 번째 li 태그만 선택)
                category_row = item.select_one('ol.foodtype li')
                
                if name_row:
                    name = name_row.text.strip()
                    address = address_row.text.strip() if address_row else ""
                    category = category_row.text.strip() if category_row else ""
                    
                    # 중복 방지 로직
                    if not any(d.get('매장명') == name for d in scraped_data):
                        scraped_data.append({
                            '매장명': name,
                            '카테고리': category,
                            '주소': address
                        })
                        
        print(f"{current_page}페이지 수집 완료. (현재 페이지 블루리본: {blue_ribbon_count_on_page}개 / 총 누적 수집: {len(scraped_data)}개)")
        
        # [종료 조건 1] 현재 페이지에 블루리본 매장이 단 하나도 없다면 종료
        if blue_ribbon_count_on_page == 0:
            print("\n[종료 알림] 더 이상 블루리본 매장이 발견되지 않아 수집을 종료합니다.")
            break
            
        # 다음 페이지로 이동
        current_page += 1
        try:
            next_button_selector = f"li[data-lp='{current_page}'] a"
            next_button = WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, next_button_selector))
            )
            
            driver.execute_script("arguments[0].scrollIntoView(true);", next_button)
            time.sleep(1) 
            driver.execute_script("arguments[0].click();", next_button)
            
            # 다음 페이지 로딩 대기 (너무 빠르면 차단될 수 있으니 3초 여유)
            time.sleep(3) 
            
        except Exception as e:
            # [종료 조건 2] 다음 페이지 버튼이 없거나 에러가 나면 종료
            print(f"\n[종료 알림] 마지막 페이지에 도달했거나 다음 페이지를 넘길 수 없습니다.")
            break

    # driver.quit()
    
    # 5. 수집된 리스트를 Pandas 데이터프레임으로 변환
    df = pd.DataFrame(scraped_data)
    return df

# 실행부
if __name__ == "__main__":
    df_results = get_all_blue_ribbon_data()
    
    print("\n[수집 완료 데이터프레임]")
    pd.set_option('display.max_rows', None)
    print(df_results.head(20)) # 너무 많을 수 있으니 화면에는 상위 20개만 출력
    print(f"\n총 {len(df_results)}개의 데이터가 수집되었습니다.")
    
    # 엑셀(CSV) 파일로 자동 저장
    file_name = "blueribbon_all_data.csv"
    df_results.to_csv(file_name, index=False, encoding='utf-8-sig')
    print(f"'{file_name}' 파일로 저장이 완료되었습니다!")