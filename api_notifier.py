# api_notifier.py

import os
import requests
import numpy as np
import time
import anomaly_handler
from dotenv import load_dotenv



def send_embedding(api_url, kiosk_id, camera_url, embedding, time_stamp, image_base):
    print("DEBUG imageB64 =>", image_base)
    payload = {
        'kiosk_id': kiosk_id,
        'camera_url': camera_url,
        'embedding': embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
        'time_stamp': time_stamp,
        'photo': image_base
    }
    response = requests.post(api_url, json=payload)
    return response.status_code, response.text

