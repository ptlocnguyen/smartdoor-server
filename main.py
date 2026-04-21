# =========================================
# SMART DOOR BACKEND - MAIN (FINAL)
# =========================================

from fastapi import FastAPI, UploadFile, File, Form
import time
import numpy as np

from face_engine import get_embedding, cosine_similarity
from cache import get_cache, refresh_cache
from config import SIM_THRESHOLD
from db import insert_log, get_connection
from typing import List

app = FastAPI()


# =========================================
# STARTUP
# =========================================
@app.on_event("startup")
def startup():
    refresh_cache()
    print("Server started - cache loaded")


# =========================================
# HEALTH CHECK
# =========================================
@app.get("/")
def health():
    return {"status": "ok"}


# =========================================
# NHẬN DIỆN KHUÔN MẶT (ESP32)
# =========================================
@app.post("/recognize")
async def recognize(file: UploadFile = File(...), device_code: str = "esp32"):

    start = time.time()

    image = await file.read()

    embedding = get_embedding(image)

    if embedding is None:
        return {"success": False, "message": "no_face"}

    cache = get_cache()

    best_user = None
    best_score = 0

    for user in cache:
        score = cosine_similarity(embedding, user["embedding"])
        if score > best_score:
            best_score = score
            best_user = user

    if best_score >= SIM_THRESHOLD:
        result = {
            "success": True,
            "user_code": best_user["user_code"],
            "full_name": best_user["full_name"],
            "similarity": float(best_score)
        }

        # ghi log
        insert_log({
            "user_id": best_user["user_id"],
            "user_code": best_user["user_code"],
            "full_name": best_user["full_name"],
            "method": "face",
            "channel": "esp32",
            "result": "success",
            "device_code": device_code,
            "similarity": best_score
        })

    else:
        result = {"success": False, "message": "unknown"}

    result["process_time"] = time.time() - start

    return result


# =========================================
# HÀM TÍNH TRUNG BÌNH EMBEDDING
# =========================================
def average_embedding(embeddings):
    return np.mean(np.array(embeddings), axis=0).tolist()


# =========================================
# API ĐĂNG KÝ KHUÔN MẶT (WEB)
# =========================================
@app.post("/register-face")
async def register_face(
    user_code: str = Form(...),
    files: List[UploadFile] = File(...)
):

    valid_embeddings = []
    failed = 0

    # =========================
    # 1. LẤY EMBEDDING TỪ ẢNH
    # =========================
    for f in files:
        img = await f.read()
        emb = get_embedding(img)

        if emb is None:
            failed += 1
        else:
            valid_embeddings.append(emb)

    if len(valid_embeddings) == 0:
        return {
            "success": False,
            "message": "no_face_detected"
        }

    # =========================
    # 2. TÍNH TRUNG BÌNH EMBEDDING MỚI
    # =========================
    new_embedding = average_embedding(valid_embeddings)

    # =========================
    # 3. KẾT NỐI DB
    # =========================
    conn = get_connection()
    cursor = conn.cursor()

    # =========================
    # 4. LẤY USER_ID
    # =========================
    cursor.execute(f"""
        SELECT user_id FROM smartdoor.core.users
        WHERE user_code = '{user_code}'
    """)

    row = cursor.fetchone()

    if not row:
        cursor.close()
        conn.close()
        return {"success": False, "message": "user_not_found"}

    user_id = row[0]

    # =========================
    # 5. CHECK EMBEDDING CŨ
    # =========================
    cursor.execute(f"""
        SELECT embedding, sample_count 
        FROM smartdoor.core.face_profiles
        WHERE user_id = {user_id}
    """)

    old = cursor.fetchone()

    if old:
        old_embedding = np.array(old[0])
        old_count = old[1]

        final_embedding = (
            (old_embedding * old_count + np.array(new_embedding))
            / (old_count + 1)
        ).tolist()

        new_count = old_count + 1

    else:
        final_embedding = new_embedding
        new_count = len(valid_embeddings)

    # =========================
    # 6. UPSERT EMBEDDING
    # =========================
    embedding_str = ",".join(map(str, final_embedding))

    cursor.execute(f"""
        MERGE INTO smartdoor.core.face_profiles AS target
        USING (
            SELECT 
                {user_id} AS user_id,
                array({embedding_str}) AS embedding,
                512 AS embedding_dim,
                {new_count} AS sample_count,
                current_timestamp() AS updated_at
        ) AS source
        ON target.user_id = source.user_id

        WHEN MATCHED THEN UPDATE SET
            embedding = source.embedding,
            sample_count = source.sample_count,
            updated_at = source.updated_at

        WHEN NOT MATCHED THEN INSERT (
            user_id, embedding, embedding_dim, sample_count, updated_at
        )
        VALUES (
            source.user_id, source.embedding, source.embedding_dim,
            source.sample_count, source.updated_at
        )
    """)

    conn.commit()

    cursor.close()
    conn.close()

    # =========================
    # 7. REFRESH CACHE
    # =========================
    refresh_cache()

    return {
        "success": True,
        "processed": len(valid_embeddings),
        "failed": failed
    }


# =========================================
# REFRESH CACHE THỦ CÔNG
# =========================================
@app.post("/refresh-cache")
def refresh():
    refresh_cache()
    return {"status": "refreshed"}
