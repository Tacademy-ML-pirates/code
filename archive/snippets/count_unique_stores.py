"""
Description:
    CSV 파일의 매장명 컬럼에서 고유 매장 수를 빠르게 확인하는 보조 코드입니다.

Author:
    이용현

Source:
    Notion ML database, page 376bf34c-c2ef-8042-ba96-e49725349a9a
"""

len(pd.read_csv("파일명.csv", encoding="utf-8-sig")["매장명"].dropna().astype(str).str.strip().unique())
