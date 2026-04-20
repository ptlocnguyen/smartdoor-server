import requests
import numpy as np
from app.db import get_connection
import os

API_URL = os.getenv("AI_URL")

def normalize(v):
    v = np.array(v, dtype=np.float32)
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def cosine(a, b):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)

    if len(a) != 512 or len(b) != 512:
        return 0.0

    a = normalize(a)
    b = normalize(b)

    return float(np.dot(a, b))


# ===== FIX EMBEDDING =====
def get_embedding(file_bytes):
    try:
        res = requests.post(
            API_URL,
            files={
                "file": ("image.jpg", file_bytes, "image/jpeg")
            },
            timeout=10
        )

        if res.status_code != 200:
            print("API ERROR:", res.text)
            return None

        data = res.json()

        if "embedding" not in data:
            print("NO EMBEDDING FIELD:", data)
            return None

        emb = data["embedding"]

        if not emb:
            print("EMPTY EMBEDDING")
            return None

        if len(emb) != 512:
            print("WRONG DIM:", len(emb))
            return None

        return emb

    except Exception as e:
        print("ERROR:", e)
        return None


def recognize_face(file_bytes):
    emb = get_embedding(file_bytes)
    if emb is None:
        return None, 0

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT user_id, avg_embedding FROM face_user_vector")

        best_user = None
        best_score = 0

        for row in cursor.fetchall():
            db_emb = row[1]

            if db_emb is None:
                continue

            try:
                db_emb = list(db_emb)
            except:
                continue

            if len(db_emb) != 512:
                continue

            score = cosine(emb, db_emb)

            if score > best_score:
                best_score = score
                best_user = row[0]

    return best_user, best_score
