import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np


# Chroma 클라이언트 초기화
chroma_client = chromadb.Client()
collection_name = 'eyewear_recommendations'

# Sentence-BERT 모델 로드
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')


def add_glasses_to_chroma(df, eyewear_collection):
    for index, row in df.iterrows():
        # 액셀 파일에서 각 행을 읽어오기
        glasses_data = row.to_dict()

        # 임베딩 생성
        embedding = generate_embedding_from_data(glasses_data)

        # 컬렉션에 벡터 추가 (임베딩과 메타데이터 추가)
        eyewear_collection.add(
            ids=[str(glasses_data['id'])],
            documents=[create_combined_text(glasses_data)],  # 안경의 설명 텍스트
            embeddings=[embedding]  # 생성된 임베딩
        )



def create_combined_text(data):
    # 안경의 속성들을 하나의 문자열로 결합
    combined_text = f"Model: {data['model']}, Color: {data['color']}, Size: {data['width']}x{data['length']} mm, Weight: {data['weight']} grams, Material: {data['material']}, Shape: {data['shape']}, Price: {data['price']}"
    return combined_text

def generate_embedding_from_data(data):
    # 결합된 텍스트로 임베딩 생성
    combined_text = create_combined_text(data)
    return model.encode(combined_text).tolist()

# Chroma 컬렉션 생성 (이미 있는 경우 불러옴)
eyewear_collection = chroma_client.get_or_create_collection(name=collection_name)

def generate_embedding(text):
    return model.encode(text).tolist()  # numpy 배열을 Chroma 호환형 리스트로 변환

# 데이터 추가 함수
def add_to_chroma_from_data(data):
    embedding = generate_embedding_from_data(data)
    eyewear_collection.add(
        documents=[create_combined_text(data)],
        metadatas=[{'model_name': data['model_name']}],
        embeddings=[embedding]
    )

def search_chroma(query, top_k=5):
    query_embedding = generate_embedding(query)
    results = eyewear_collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results

# 예시 매핑 테이블
mapping_table = [
    {'face_shape': '둥근형', 'skin_tone': '웜톤', 'recommended_colors': ['brown', 'gold'], 'recommended_shapes': ['square', 'poly']},
    {'face_shape': '네모형', 'skin_tone': '쿨톤', 'recommended_colors': ['silver', 'navy'], 'recommended_shapes': ['round']},
]


def get_suggested_glasses(user_face_shape, user_skin_tone):
    suggested_glasses = []

    for entry in mapping_table:
        if entry['face_shape'] == user_face_shape and entry['skin_tone'] == user_skin_tone:
            suggested_glasses.append(entry)

    return suggested_glasses





# Chroma에서 필터링된 색상과 형태에 맞는 안경을 찾기
def filter_glasses_by_mapping(suggested_glasses, eyewear_collection):
    filtered_glasses = []

    for glasses in suggested_glasses:
        # 색상 필터링
        for color in glasses['recommended_colors']:
            color_embedding = model.encode(color).tolist()  # 색상을 임베딩으로 변환
            color_results = eyewear_collection.query(query_embeddings=[color_embedding], n_results=10)
            filtered_glasses += color_results["documents"]

        # 형태 필터링
        for shape in glasses['recommended_shapes']:
            shape_embedding = model.encode(shape).tolist()  # 형태를 임베딩으로 변환
            shape_results = eyewear_collection.query(query_embeddings=[shape_embedding], n_results=10)
            filtered_glasses += shape_results["documents"]
    return filtered_glasses


def vector_search_in_chroma(filtered_glasses, eyewear_collection):
    results = []

    for glasses in filtered_glasses:
        query_embedding = model.encode(create_combined_text(glasses))  # 각 안경의 임베딩을 생성
        result = eyewear_collection.query(query_embeddings=[query_embedding], n_results=10)
        filtered_result_ids = [doc['id'] for doc in result['documents']]
        results.extend(filtered_result_ids)

    return results
