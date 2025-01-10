import glob
import json
import random

# 데이터 디렉토리 경로
DATA_DIR = "/Users/hong-gihyeon/Desktop/NotiFYI_BackEND/app/data"

def load_and_shuffle_data():
    """
    JSON 파일에서 데이터를 로드하고 랜덤으로 섞습니다.
    """
    all_data = []
    
    # 디렉토리 내 모든 JSON 파일 읽기
    for file_path in glob.glob(f"{DATA_DIR}/*.json"):
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            # 데이터가 리스트인지 확인 후 병합
            if isinstance(data, list):
                all_data.extend(data)
            else:
                all_data.append(data)
    
    # 데이터 섞기
    random.shuffle(all_data)
    return all_data