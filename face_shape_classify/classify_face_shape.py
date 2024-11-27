import numpy as np


# 얼굴형 분류 함수
def classify_face_shape(landmarks):
    # 턱 라인의 좌표
    jaw_points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(0, 17)])
    # 얼굴 높이 측정 (턱에서 미간까지)
    face_height = np.linalg.norm(np.array([landmarks.part(8).x, landmarks.part(8).y]) - np.array([landmarks.part(27).x, landmarks.part(27).y]))  # 턱에서 이마 중심까지
    # 얼굴 너비 측정 (광대뼈 간 거리)
    face_width = np.linalg.norm(np.array([landmarks.part(0).x, landmarks.part(0).y]) - np.array([landmarks.part(16).x, landmarks.part(16).y]))
    # 턱 너비 측정 (턱 가장자리 간 거리)
    jaw_width = np.linalg.norm(np.array([landmarks.part(5).x, landmarks.part(5).y]) - np.array([landmarks.part(11).x, landmarks.part(11).y]))

    # 얼굴형 비율 계산
    jaw_ratio = jaw_width / face_width
    face_ratio = face_width / (face_height*1.5)


    # 얼굴형 분류 기준 적용
    if face_ratio < 0.75 and 0.575 < jaw_ratio < 0.75:
        return "타원형"
    elif 0.75 <= face_ratio and 0.575 < jaw_ratio < 0.75:
        return "각진형"
    elif 0.75 <= face_ratio and 0.75 <= jaw_ratio:
        return "둥근형"
    elif 0.75 <= face_ratio and jaw_ratio <= 0.575:
        return "하트형"
    elif face_ratio < 0.75 and 0.75 <= jaw_ratio:
        return "긴형"
    elif face_ratio < 0.75 and jaw_ratio <= 0.575:
        return "다이아형"
    else:
        return "Unknown"

# Oval: 길고 둥근 얼굴. 이마와 턱은 좁고, 윤곽이 둥글며, 얼굴이 긺.
# Round: 원형 얼굴. 광대뼈가 넓고 얼굴 길이가 길지 않으며 턱선이 부드러움.
# Heart: 하트형 얼굴. 이마가 넓고 턱은 좁으며 얼굴 길이는 길지 않음.
# Square: 사각형 얼굴. 이마와 턱이 각지고 얼굴 너비는 넓고 얼굴 길이는 길지 않음.
# Oblong: 길고 각진 얼굴. 이마와 턱이 넓고, 윤곽이 각지며, 얼굴이 긺.
# Diamond: 다이아몬드형 얼굴. 이마와 턱이 좁고 얼굴 길이는 긴 편이며 광대가 두드러짐.
