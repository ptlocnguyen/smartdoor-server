import requests
import numpy as np
from app.db import get_connection
import os

API_URL = os.getenv("AI_URL")

def normalize(v):
    v = np.array(v, dtype=np.float32)
    norm = np.linalg.norm(v)
    return v if norm == 0 else v / norm


def cosine(a, b):
    if len(a) != 512 or len(b) != 512:
        return 0.0

    a = normalize(a)
    b = normalize(b)

    return float(np.dot(a, b))


# ===== FIX EMBEDDING + NHẸ =====
def get_embedding(file_bytes):
    try:
        res = requests.post(
            API_URL,
            files={"file": ("img.jpg", file_bytes, "image/jpeg")},
            timeout=8  # GIẢM TIMEOUT
        )

        if res.status_code != 200:
            return None

        data = res.json()

        emb = data.get("embedding")

        if not emb or len(emb) != 512:
            return None

        return emb

    except Exception:
        return None


def recognize_face(file_bytes):
    emb = get_embedding(file_bytes)
    if emb is None:
        return None, 0

    with get_connection() as conn:
        cursor = conn.cursor()

        # LIMIT để tránh nặng
        cursor.execute("""
            SELECT user_id, avg_embedding
            FROM face_user_vector
            LIMIT 50
        """)

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
