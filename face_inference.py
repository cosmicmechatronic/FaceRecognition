# face_inference.py
import anomaly_handler
import numpy as np
import cv2

from mtcnn_client import MtCnnClient
from bounding_box import BoundingBox
from utils import resize_image, normalize_input

class FaceInference:
    def __init__(self, face_model, model_info):
        self.face_model = face_model
        self.model_info = model_info
        self.detector = MtCnnClient()

    def process_image(self, img_rgb: np.ndarray):
        """ Wykrywa twarze przez MTCNN, wycina je, zwraca listę (face_img, bbox). """
        detections = self.detector.detect_faces(img_rgb)
        faces_info = []

        for det in detections:
            bbox = det['bbox']
            face_region = self.extract_face(img_rgb, bbox)
            if face_region is not None and face_region.size != 0:
                faces_info.append((face_region, bbox))
            else:
                # Logujemy, że bounding box był nieprawidłowy lub dał pusty obraz
                anomaly_handler.log_warning("Otrzymano pusty wycinek twarzy; pomijam.")

        return faces_info

    def extract_face(self, img_rgb: np.ndarray, bbox: BoundingBox):
        """ Wycinamy fragment (twarz) z obrazu, z uwzględnieniem granic. """
        x1, y1, x2, y2 = bbox.to_xyxy()

        # Docięcie do wymiarów obrazu
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_rgb.shape[1], x2)
        y2 = min(img_rgb.shape[0], y2)

        if x2 <= x1 or y2 <= y1:
            return None

        face_region = img_rgb[y1:y2, x1:x2]
        if face_region.size == 0:
            return None
        return face_region

    def compute_embedding(self, face_img: np.ndarray):
        """ 
        Oblicza embedding. Zwraca None, jeśli otrzyma pusty obraz.
        Upewniamy się, że tylko RAZ dodajemy wymiar batchu.
        """
        if face_img is None or face_img.size == 0:
            return None

        # Resize => (160,160)
        face_input = cv2.resize(face_img, (160, 160))
        face_input = normalize_input(face_input, "base")

        # Dodaj batch dimension TYLKO RAZ => (1,160,160,3)
        face_input = np.expand_dims(face_input, axis=0)
        
        try:
            out = self.face_model(face_input, training=False)
        except Exception as e:
            anomaly_handler.log_error(f"Błąd w obliczaniu embeddingu: {str(e)}")
            return None
        
        embedding = out.numpy()[0]  # shape => (128,) np.
        return embedding
