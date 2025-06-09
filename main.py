# main.py

import os
import time
import requests
from requests.auth import HTTPBasicAuth
import base64
import cv2

from dotenv import load_dotenv

import anomaly_handler
from video_reader import VideoReader
from mtcnn_client import MtCnnClient
from bounding_box import BoundingBox
from facenet import InceptionResNetV1
from face_inference import FaceInference
from api_notifier import send_embedding

from local_verification import store_local_data


def main():
    """
    Główna pętla aplikacji (skrótowa wersja).
    1) Pobiera z .env m.in. PARAM_WIDTH, PARAM_HEIGHT.
    2) Po wykryciu ruchu, czyta klatkę z kamery, wykrywa twarz, 
       sprawdza czy bounding box >= param_width i param_height, itd.
    """
    load_dotenv()
    anomaly_handler.log_info("=== Start aplikacji ===")

    # Wczytujemy parametry z .env
    sensor_url = os.environ.get("PROXIMITY_SENSOR_URL")
    camera_url = os.environ.get("CAMERA_URL")
    camera_user = os.environ.get("CAMERA_USERNAME")
    camera_pass = os.environ.get("CAMERA_PASSWORD")

    api_url  = os.environ.get("API_URL")
    kiosk_id = os.environ.get("KIOSK_ID", "1")

    cold_mode = float(os.environ.get("COLD_MODE", 1.0))
    hot_mode  = float(os.environ.get("HOT_MODE", 0.8))

    # NOWE: wczytujemy parametry sprawdzania bounding boxa
    parameter_width  = float(os.environ.get("PARAM_WIDTH", 0.25))   # np. 0.25
    parameter_height = float(os.environ.get("PARAM_HEIGHT", 0.25))  # np. 0.25

    anomaly_handler.log_info(
        f"Wczytano parametry: PARAM_WIDTH={parameter_width}, PARAM_HEIGHT={parameter_height}"
    )

    # Inicjalizacja modelu FaceNet
    anomaly_handler.log_info("Ładowanie modelu FaceNet (model.h5)...")
    face_model = InceptionResNetV1(dimension=128)
    face_model.load_weights("model.h5")

    # Tworzymy obiekt FaceInference (wykorzysta MTCNN + FaceNet)
    inference_class = FaceInference(
        face_model=face_model,
        model_info={"framework": "tf", "model": "facenet", "dimension": 128}
    )

    # Inicjalizacja strumienia z kamery
    try:
        video_reader = VideoReader(camera_url, camera_user, camera_pass)
    except ValueError:
        anomaly_handler.camera_connection_error(camera_url)
        return

    # Zmienne sterujące pętlą
    face_detected = False
    mode_interval = cold_mode
    last_check_time = time.time()

    anomaly_handler.log_info("=== Aplikacja ruszyła w pętli głównej ===")

    while True:
        now = time.time()
        if (now - last_check_time) >= mode_interval:
            last_check_time = now

            # Przykładowe sprawdzenie czujnika:
            detection = check_sensor(sensor_url, camera_user, camera_pass)
            if not detection:
                anomaly_handler.log_info("Brak ruchu - czekam...")
                face_detected = False
                mode_interval = cold_mode
                continue

            # Mamy ruch, więc pobieramy klatkę z kamery
            frame_rgb, capture_time = video_reader.read_frame()
            if frame_rgb is None:
                anomaly_handler.log_warning("Brak klatki z kamery.")
                continue

            # Wykrycie twarzy (lista (face_img, bbox))
            faces_info = inference_class.process_image(frame_rgb)
            if not faces_info:
                # Jeśli nie znaleziono twarzy -> hot_mode
                if face_detected:
                    anomaly_handler.log_info("Twarz zniknęła, przechodzę do hot_mode.")
                face_detected = False
                mode_interval = hot_mode
                continue

            # Bierzemy pierwszą twarz:
            (face_img, bbox) = faces_info[0]

            # Tu sprawdzamy minimalny rozmiar bounding boxa względem całego kadru
            h_frame, w_frame, _ = frame_rgb.shape
            if (bbox.width < parameter_width  * w_frame or
                bbox.height < parameter_height * h_frame):
                anomaly_handler.log_info("Twarz za mała. Ustawiam hot_mode.")
                face_detected = False
                mode_interval = hot_mode
                continue

            # Mamy wystarczająco dużą twarz => obliczamy embedding itp.
            face_detected = True
            mode_interval = cold_mode

            # (reszta logiki: rysowanie prostokąta, obliczanie embedding, wysyłka do API itd.)
            embedding = inference_class.compute_embedding(face_img)
            if embedding is None:
                anomaly_handler.log_warning("Embedding nie został wyliczony.")
                continue

            # Zamiana całej klatki na base64
            frame_bgr = frame_rgb[:, :, ::-1]
            success, buffer = cv2.imencode('.jpg', frame_bgr)
            if not success:
                anomaly_handler.log_warning("Nie udało się zakodować klatki do JPEG.")
                continue

            image_base = base64.b64encode(buffer).decode('utf-8')

            if image_base is None:
                print("image_base jest None – obraz nie został zakodowany!")
            else:
                if len(image_base) == 0:
                    print("image_base jest pusty (ciąg o długości 0)!")
                else:
                    print(f"OK, image_base ma długość: {len(image_base)}")

            # Zapis lokalnie, wysyłka do API, itd.
            
            status, resp = send_embedding(api_url, kiosk_id, camera_url, embedding, capture_time, image_base)
            
            anomaly_handler.log_info(f"Wynik zapisu w API: status={status}, response={resp}")
            # Wywołanie funkcji zapisu na dysk
            
            store_local_data(image_base, status, resp)

        time.sleep(0.05)


def check_sensor(sensor_url, username=None, password=None):
    """ Przykładowe sprawdzanie stanu czujnika. Zwraca True/False. """
    if not sensor_url:
        anomaly_handler.log_warning("Brak sensor_url")
        return False
    try:
        r = requests.get(sensor_url, auth=HTTPBasicAuth(username, password), timeout=3)
        r.raise_for_status()
        data = r.json()
        value = data.get('LL', {}).get('value')
        if value == "1":
            anomaly_handler.log_info("Czujnik => wykryto ruch (value=1).")
            return True
        else:
            anomaly_handler.log_info("Czujnik => brak ruchu (value=0).")
            return False
    except requests.RequestException as e:
        anomaly_handler.sensor_connection_error(sensor_url)
        return False

if __name__ == "__main__":
    main()
