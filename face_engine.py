import requests
import numpy as np
from config import BUFFALO_API_URL

def get_embedding(image_bytes):
    try:
        resp = requests.post(
            BUFFALO_API_URL,
            files={"file": ("image.jpg", image_bytes, "image/jpeg")},
            timeout=10
        )
        if resp.status_code == 200:
            data = resp.json()
            return data.get("embedding")
    except:
        return None

    return None


def cosine_similarity(a, b):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)

    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
