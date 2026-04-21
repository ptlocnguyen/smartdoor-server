# =========================================
# SMART DOOR BACKEND - FINAL MAIN
# =========================================

from fastapi import FastAPI, UploadFile, File, Form
from typing import List
import time
import numpy as np

from face_engine import get_embedding, cosine_similarity
from cache import get_cache, refresh_cache
from config import SIM_THRESHOLD
from db import insert_log, get_connection

from fastapi.openapi.utils import get_openapi

app = FastAPI()


# =========================================
# FIX SWAGGER FILE UPLOAD (QUAN TRỌNG)
# =========================================
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="SmartDoor API",
        version="1.0.0",
        description="API",
        routes=app.routes,
    )

    # Ép kiểu file upload
    try:
        schema = openapi_schema["paths"]["/register-face"]["post"]["requestBody"]["content"]["multipart/form-data"]["schema"]
        schema["properties"]["files"] = {
            "type": "array",
            "items": {
                "type": "string",
                "format": "binary"
            }
        }
    except Exception as e:
        print("OpenAPI fix error:", e)

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


# =========================================
# STARTUP
# =========================================
@app.on_event("startup")
def startup():
    refresh_cache()
    print("Server started - cache loaded")


# =========================================
# HEALTH
# =========================================
@app.get("/")
def health():
    return {"status": "ok"}


# =========================================
# RECOGNIZE FACE (ESP32)
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
        if len(user["embedding"]) != len(embedding):
            continue

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
# AVERAGE EMBEDDING
# =========================================
def average_embedding(embeddings):
    return np.mean(np.array(embeddings), axis=0)


# =========================================
# REGISTER FACE
# =========================================
@app.post("/register-face")
async def register_face(
    user_code: str = Form(...),
    files: List[UploadFile] = File(...)
):

    valid_embeddings = []
    failed = 0

    # =========================
    # 1. LẤY EMBEDDING
    # =========================
    for f in files:
        img = await f.read()
        emb = get_embedding(img)

        if emb is None or len(emb) != 512:
            failed += 1
        else:
            valid_embeddings.append(emb)

    if len(valid_embeddings) == 0:
        return {"success": False, "message": "no_face_detected"}

    new_embedding = average_embedding(valid_embeddings)

    conn = get_connection()
    cursor = conn.cursor()

    # =========================
    # 2. LẤY USER_ID (ANTI SQL INJECTION)
    # =========================
    cursor.execute(
        "SELECT user_id FROM smartdoor.core.users WHERE user_code = ?",
        (user_code,)
    )

    row = cursor.fetchone()

    if not row:
        cursor.close()
        conn.close()
        return {"success": False, "message": "user_not_found"}

    user_id = row[0]

    # =========================
    # 3. LẤY EMBEDDING CŨ
    # =========================
    cursor.execute(
        "SELECT embedding, sample_count FROM smartdoor.core.face_profiles WHERE user_id = ?",
        (user_id,)
    )

    old = cursor.fetchone()

    if old:
        old_embedding = np.array(old[0])
        old_count = old[1]

        total_count = old_count + len(valid_embeddings)

        final_embedding = (
            old_embedding * old_count +
            np.array(new_embedding) * len(valid_embeddings)
        ) / total_count

        new_count = total_count

    else:
        final_embedding = np.array(new_embedding)
        new_count = len(valid_embeddings)

    embedding_str = ",".join(map(str, final_embedding.tolist()))

    # =========================
    # 4. UPSERT
    # =========================
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
    # 5. REFRESH CACHE
    # =========================
    refresh_cache()

    return {
        "success": True,
        "processed": len(valid_embeddings),
        "failed": failed
    }


# =========================================
# REFRESH CACHE
# =========================================
@app.post("/refresh-cache")
def refresh():
    refresh_cache()
    return {"status": "refreshed"}
