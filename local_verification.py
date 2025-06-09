import os
import time
import base64
import json

def store_local_data(image_base64: str, server_status: int, server_response: str):
    """
    Zapisuje na dysk plik JPEG oraz plik JSON (odpowiedź serwera).
    
    :param image_base64: Obraz w formie base64.
    :param server_status: Kod statusu HTTP z odpowiedzi serwera.
    :param server_response: Treść odpowiedzi serwera (zwykle JSON w formie stringa).
    """
    # Upewnij się, że istnieje folder 'stored_data'
    if not os.path.exists("stored_data"):
        os.makedirs("stored_data")

    # Utworzenie nazwy plików na bazie stempla czasowego
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")

    # 1) Zapis pliku JPEG
    try:
        image_data = base64.b64decode(image_base64)
        jpg_filename = f"{timestamp_str}.jpg"
        with open(os.path.join("stored_data", jpg_filename), "wb") as f:
            f.write(image_data)
    except Exception as e:
        print(f"Błąd podczas zapisu pliku JPEG: {e}")

    # 2) Zapis odpowiedzi serwera w formacie JSON
    #    Odpowiedź może być już w formacie JSON-owym lub zwykłym tekstem –
    #    w zależności od tego, co zwraca API. Na wszelki wypadek serializujemy
    #    całość jako klucz w dict.
    try:
        json_filename = f"{timestamp_str}.json"
        data_to_save = {
            "status": server_status,
            "response": json.loads(server_response)
        }
        with open(os.path.join("stored_data", json_filename), "w", encoding='utf-8') as jf:
            json.dump(data_to_save, jf, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Błąd podczas zapisu pliku JSON: {e}")
