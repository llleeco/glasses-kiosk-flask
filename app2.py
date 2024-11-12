from flask import Flask, request, render_template, jsonify
import cv2
import numpy as np
import base64
from PIL import Image
import io
import time
from personal_color_analysis import personal_color

app = Flask(__name__)

#웹 페이지 렌더링
@app.route('/')
def index():
    return render_template('index.html')

# 사진 업로드 및 얼굴 탐지 및 피부톤 추출
@app.route('/upload', methods=['POST'])
def upload():
    files = request.files.getlist('image')
    if files:
        file = files[0]
        image_data = file.read()
        image = Image.open(io.BytesIO(image_data))

        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
  
        # 이미지 저장 후 경로 전달
        img_path = './uploaded_image.jpg'
        cv2.imwrite(img_path, img_cv)
        # 피부톤 분석
        tone, skin_tone = personal_color.analysis(img_cv)
        
        # skin_tone을 int32에서 기본 int로 변환
        skin_tone = [int(c) for c in skin_tone]  # numpy int32 -> int 변환
        
        return jsonify({
            "tone": tone,
            "skin_tone": skin_tone
        })
    else:
        return jsonify({"error": "No image uploaded"}), 400

if __name__ == "__main__":
    app.run('0.0.0.0',debug=True)
