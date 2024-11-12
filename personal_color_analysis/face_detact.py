import dlib
import cv2


class DetectFace:
    def __init__(self, img_cv):
        self.detector = dlib.get_frontal_face_detector()
        self.img = img_cv
        self.face_img = None
        self.detect_face()

    def detect_face(self):
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            
        detections = self.detector(gray, 1)
        if detections:
            for detection in detections:
                x, y, w, h = detection.left(), detection.top(), detection.width(), detection.height()
                self.face_img = self.img[y:y+h, x:x+w]
        return self.face_img
