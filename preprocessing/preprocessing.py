import cv2
import dlib
import imutils
from imutils import face_utils
from ultralytics import YOLO
import math

class Preprocessor:
    def __init__(self, yolo_model, dlib_model, target_size=224):
        self.detect = YOLO(yolo_model)
        self.predictor = dlib.shape_predictor(dlib_model)
        self.target_size = target_size

    def detect_faces(self, image):
        results = self.detect(image)
        boxes = []
        for result in results:
            for box in result.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box)
                boxes.append((x1, y1, x2, y2))
        return boxes

    def align_face(self, image, box):
        x1, y1, x2, y2 = box
        rect = dlib.rectangle(x1, y1, x2, y2)
        shape = self.predictor(image, rect)
        shape = face_utils.shape_to_np(shape)
        r_eye = shape[36]
        l_eye = shape[45]
        dx = r_eye[0] - l_eye[0]
        dy = r_eye[1] - l_eye[1]
        alpha = math.degrees(math.atan2(dy, dx))
        cropped = image[y1:y2, x1:x2].copy()
        cropped = imutils.rotate(cropped, 180 + alpha)
        return cv2.resize(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB), (224, 224))

    def preprocess(self, image_path):
        image = cv2.imread(image_path)
        faces = []
        boxes = self.detect_faces(image)

        for box in boxes:
            aligned = self.align_face(image, box)
            faces.append(aligned)
        return faces