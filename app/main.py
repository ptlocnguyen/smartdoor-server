from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from app.face import recognize_face, get_embedding
from app.db import get_connection
import uuid
import numpy as np

app = FastAPI()

# ===== CORS =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===== USER =====
@app.post("/users/create")
def create_user(name: str, phone: str = "", role: str = "user"):
    user_id = str(uuid.uuid4())

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO users VALUES (?, ?, ?, ?, current_timestamp())",
            (user_id, name, phone, role)
        )

    return {"status": "ok", "user_id": user_id}


@app.get("/users")
def get_users():
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT user_id, name FROM users")

        return [{"user_id": r[0], "name": r[1]} for r in cursor.fetchall()]


# ===== REGISTER FACE =====
@app.post("/face/register")
async def register_face(user_id: str, file: UploadFile = File(...)):
    file_bytes = await file.read()
    emb = get_embedding(file_bytes)

    if emb is None:
        return {"error": "Embedding failed"}

    with get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
        if not cursor.fetchone():
            return {"error": "User not found"}

        sample_id = str(uuid.uuid4())

        cursor.execute(
            "INSERT INTO face_samples VALUES (?, ?, ?, current_timestamp())",
            (sample_id, user_id, emb)
        )

        # ===== LOAD ALL EMBEDDING =====
        cursor.execute(
            "SELECT embedding FROM face_samples WHERE user_id = ?",
            (user_id,)
        )

        rows = cursor.fetchall()

        valid = []

        for r in rows:
            emb_db = r[0]

            if emb_db is None:
                continue

            try:
                emb_list = list(emb_db)
            except:
                continue

            if len(emb_list) == 512:
                valid.append(emb_list)

        if len(valid) == 0:
            return {"error": "No valid embedding"}

        avg = np.mean(valid, axis=0).tolist()

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


# ===== RECOGNIZE =====
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


@app.get("/")
def root():
    return {"status": "running"}
