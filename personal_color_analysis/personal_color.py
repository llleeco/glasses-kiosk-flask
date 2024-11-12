import numpy as np
from personal_color_analysis import tone_analysis
from personal_color_analysis.face_detact import DetectFace
from personal_color_analysis.color_extract import DominantColors
from colormath.color_objects import LabColor, sRGBColor, HSVColor
from colormath.color_conversions import convert_color

def analysis(imgpath):
    #face detection
    df = DetectFace(imgpath)
    face_img = df.face_img
    
    #피부 색상 추출
    if face_img is not None:
        dc = DominantColors(face_img, clusters=3)
       # skin_color = dc.getHistogram()[0]  # 첫 번째 색상만 사용

        skin_color = dc.getDominantColor()  # 단일 색상 배열

        # RGB 값을 정수로 변환하여 sRGBColor에 전달
        r, g, b = skin_color[0], skin_color[1], skin_color[2]
        rgb = sRGBColor(r, g, b, is_upscaled=True)
        lab = convert_color(rgb, LabColor)
        hsv = convert_color(rgb, HSVColor)

        Lab_b = [lab.lab_b]
        hsv_s = [hsv.hsv_s * 100]

            # 디버그 출력: 값 확인
        print("Lab_b:", Lab_b)
        print("hsv_s:", hsv_s)

        personal_color = classify_personal_color(Lab_b, hsv_s)
        print("퍼스널컬러:", personal_color)  # 퍼스널컬러 출력

        # skin_tone을 RGB 배열로 반환
        skin_tone = [r, g, b]  # 피부 색상 (RGB)
        
        return personal_color, skin_tone
    else:
        return "얼굴을 찾을 수 없습니다."
    
def classify_personal_color(Lab_b, hsv_s):
    Lab_weight = [30]
    hsv_weight = [10]
    
# Lab_b와 hsv_s의 길이가 맞는지 확인하는 조건문 추가
    if len(Lab_b) == 1 and len(hsv_s) == 1:
        if tone_analysis.is_warm(Lab_b, Lab_weight):
            return '봄웜톤' if tone_analysis.is_spr(hsv_s, hsv_weight) else '가을웜톤'
        else:
            return '여름쿨톤' if tone_analysis.is_smr(hsv_s, hsv_weight) else '겨울쿨톤'
    else:
        # Lab_b나 hsv_s의 길이가 예상과 다르면 오류 메시지 반환
        return "색상 분석 오류: Lab_b 또는 hsv_s의 길이가 올바르지 않습니다."

