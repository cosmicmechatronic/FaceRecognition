import logging
import os
from logging.handlers import RotatingFileHandler

# Ustalanie nazwy pliku logów
log_filename = "app.log"

# Konfiguracja loggera z ograniczeniem wielkości do 2 MB (2 * 1024 * 1024 bajtów)
# backupCount=5 oznacza, że będzie utrzymywanych do 5 zarchiwizowanych plików logów
handler = RotatingFileHandler(
    log_filename,
    maxBytes=2*1024*1024,
    backupCount=5,
    encoding="utf-8"
)

formatter = logging.Formatter(
    fmt='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

handler.setFormatter(formatter)
logger = logging.getLogger("AnomalyHandler")
logger.setLevel(logging.INFO)
logger.addHandler(handler)

# Dodanie handlera dla konsoli
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def log_info(message):
    logger.info(message)

def log_warning(message):
    logger.warning(message)

def log_error(message):
    logger.error(message, exc_info=True)

def handle_anomaly(anomaly_message):
    # Ogólna obsługa anomalii
    logger.error(f"Anomalia wykryta: {anomaly_message}", exc_info=True)

### Funkcje logujące specyficzne problemy ###
def camera_connection_error(camera_url):
    message = f"Brak połączenia z kamerą: {camera_url}"
    logger.error(message, exc_info=True)

def api_connection_error(api_url):
    message = f"Brak połączenia z API: {api_url}"
    logger.error(message, exc_info=True)

def face_detection_error():
    message = "Problemy z rozpoznaniem twarzy przez MTCNN."
    logger.error(message, exc_info=True)

def embedding_error():
    message = "Problemy z wygenerowaniem embeddingu twarzy."
    logger.error(message, exc_info=True)

### Dodatkowe funkcje dla czujnika ###
def sensor_connection_error(sensor_url):
    message = f"Brak połączenia z czujnikiem: {sensor_url}"
    logger.error(message, exc_info=True)

def sensor_state_change(state, value):
    message = f"Zmiana stanu czujnika: {state} (wartość: {value})"
    logger.info(message)

def worker_state_change(state):
    message = f"Zmiana stanu workera: {state}"
    logger.info(message)

def config_error(config_type, error):
    message = f"Błąd konfiguracji {config_type}: {error}"
    logger.error(message, exc_info=True)  


# --- NOWE FUNKCJE DLA STANÓW ---
def person_detected():
    logger.info("Rozpoznano osobę")

def face_recognized():
    logger.info("Rozpoznano twarz (bounding box >= 25% kadru)")

def embedding_sent():
    logger.info("Wysłano dane do API")