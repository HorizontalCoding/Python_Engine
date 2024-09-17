import requests
import difflib

visit_area_nm = '쏠비치 삼척'
csv_road_nm_addr = '강원 삼척시 수로부인길 453'
csv_lotno_addr = '강원 삼척시 갈천동 225'
important_keywords = ['쏠비치', '호텔&리조트', '삼척']  # 중요한 키워드 목록

api_key = 'wNdz4am8tX0DttDn4qesUPuL+I7E/3pOZIZOQom6SPjkWu6Fzvbqv6O4b1yoC8oeqxVWfYIB0pb4kJaF/0f/qA=='
endpoint = 'http://api.visitkorea.or.kr/openapi/service/rest/KorService/searchKeyword'
params = {
    'ServiceKey': api_key,
    'numOfRows': 1,
    'pageNo': 1,
    'MobileOS': 'ETC',
    'MobileApp': 'AppName',
    'keyword': visit_area_nm,
    'listYN': 'Y',
    'arrange': 'E',
    'areaCode': '32',
    '_type': 'json'
}

response = requests.get(endpoint, params=params)

def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):].strip()
    return text

def keyword_weighted_similarity(title, keywords):
    """키워드 가중치를 부여한 유사도 계산"""
    similarity = difflib.SequenceMatcher(None, visit_area_nm, title).ratio()
    for keyword in keywords:
        if keyword in title:
            similarity += 0.1  # 각 키워드에 가중치를 부여
    return min(similarity, 1.0)  # 유사도는 1.0을 넘지 않도록

if response.status_code == 200:
    data = response.json()
    if 'item' in data['response']['body']['items']:
        max_similarity = 0
        best_match = None
        for item in data['response']['body']['items']['item']:
            addr1 = item.get('addr1', '')
            addr2 = item.get('addr2', '')
            title = item.get('title', '')
            
            # '강원특별자치도' 제거 후 비교
            addr1_clean = remove_prefix(addr1, '강원특별자치도')
            addr2_clean = remove_prefix(addr2, '강원특별자치도')
            
            similarities = []
            
            # 도로명 주소 유사도 계산
            if csv_road_nm_addr:
                road_nm_similarity = difflib.SequenceMatcher(None, csv_road_nm_addr, addr1_clean).ratio()
                similarities.append(road_nm_similarity)
            
            # 지번 주소 유사도 계산
            if csv_lotno_addr:
                lotno_similarity = difflib.SequenceMatcher(None, csv_lotno_addr, addr2_clean).ratio()
                similarities.append(lotno_similarity)
            
            # 키워드 가중치가 적용된 이름 유사도 계산
            title_similarity = keyword_weighted_similarity(title, important_keywords)
            similarities.append(title_similarity)
            
            # 종합 유사도 계산 (유사도의 평균을 사용할 수 있음)
            if similarities:
                avg_similarity = sum(similarities) / len(similarities)
                if avg_similarity > max_similarity:
                    max_similarity = avg_similarity
                    best_match = item
        
        if best_match:
            print(f"Best Match Found: {best_match['title']}, Address: {best_match['addr1']} {best_match['addr2']}")
            print(f"Similarity: {max_similarity:.2f}")
else:
    print(f"Error: Received status code {response.status_code}")
