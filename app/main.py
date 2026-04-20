from fastapi import FastAPI, UploadFile, File
from app.face import recognize_face, get_embedding
from app.db import get_connection
from app.fingerprint import sync_fingerprint
import uuid
import numpy as np

app = FastAPI()

# ================= USER =================
@app.post("/users/create")
def create_user(name: str, phone: str = "", role: str = "user"):
    user_id = str(uuid.uuid4())

    with get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO users VALUES (?, ?, ?, ?, current_timestamp())",
            (user_id, name, phone, role)
        )

    return {
        "status": "ok",
        "user_id": user_id
    }


@app.get("/users")
def get_users():
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT user_id, name FROM users")

        return [
            {"user_id": r[0], "name": r[1]}
            for r in cursor.fetchall()
        ]


# ================= FACE RECOGNIZE =================
@app.post("/face/recognize")
async def face_recognize(file: UploadFile = File(...)):
    file_bytes = await file.read()

    user_id, score = recognize_face(file_bytes)
    result = "success" if user_id and score > 0.5 else "fail"

    with get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO access_logs VALUES (?, ?, ?, ?, ?, ?, current_timestamp())",
            (
                str(uuid.uuid4()),
                user_id,
                "face",
                "esp32",
                score,
                result
            )
        )

    return {
        "user_id": user_id,
        "score": score,
        "result": result
    }


# ================= FACE REGISTER =================
@app.post("/face/register")
async def register_face(user_id: str, file: UploadFile = File(...)):
    file_bytes = await file.read()
    emb = get_embedding(file_bytes)

    if emb is None:
        return {"error": "No face detected"}

    with get_connection() as conn:
        cursor = conn.cursor()

        # check user
        cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
        if not cursor.fetchone():
            return {"error": "User not found"}

        sample_id = str(uuid.uuid4())

        cursor.execute(
            "INSERT INTO face_samples VALUES (?, ?, ?, current_timestamp())",
            (sample_id, user_id, emb)
        )

        # update avg embedding
        cursor.execute(
            "SELECT embedding FROM face_samples WHERE user_id = ?",
            (user_id,)
        )

        rows = cursor.fetchall()
        avg = np.mean([r[0] for r in rows], axis=0).tolist()

        cursor.execute(
            """
            MERGE INTO face_user_vector t
            USING (SELECT ? as user_id, ? as emb) s
            ON t.user_id = s.user_id
            WHEN MATCHED THEN
              UPDATE SET avg_embedding = s.emb, updated_at = current_timestamp()
            WHEN NOT MATCHED THEN
              INSERT (user_id, avg_embedding, updated_at)
              VALUES (s.user_id, s.emb, current_timestamp())
            """,
            (user_id, avg)
        )

    return {"status": "ok"}


# ================= FINGERPRINT =================
@app.post("/fingerprint/sync")
def api_sync_fingerprint(data: dict):
    return sync_fingerprint(
        data["user_id"],
        data["fingerprint_id"],
        data["device_id"]
    )


# ================= LOG =================
@app.get("/logs")
def get_logs():
    with get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT user_id, method, result, created_at
            FROM access_logs
            ORDER BY created_at DESC
            LIMIT 50
        """)

        return [
            {
                "user_id": r[0],
                "method": r[1],
                "result": r[2],
                "time": str(r[3])
            }
            for r in cursor.fetchall()
        ]


@app.get("/")
def root():
    return {"status": "running"}
