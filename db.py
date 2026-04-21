from databricks import sql
from config import DB_HOST, DB_TOKEN, DB_HTTP_PATH

def get_connection():
    return sql.connect(
        server_hostname=DB_HOST,
        http_path=DB_HTTP_PATH,
        access_token=DB_TOKEN
    )

def load_face_cache():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT user_id, user_code, full_name, embedding
        FROM smartdoor.core.face_cache
    """)

    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    result = []
    for r in rows:
        result.append({
            "user_id": r[0],
            "user_code": r[1],
            "full_name": r[2],
            "embedding": r[3]
        })

    return result

def insert_log(data):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(f"""
        INSERT INTO smartdoor.core.access_logs (
            user_id,
            user_code,
            full_name_snapshot,
            identify_method,
            channel,
            result,
            device_code,
            similarity,
            captured_at_utc,
            created_at_utc
        ) VALUES (
            {data.get("user_id", "NULL")},
            '{data.get("user_code", "")}',
            '{data.get("full_name", "")}',
            '{data.get("method")}',
            '{data.get("channel")}',
            '{data.get("result")}',
            '{data.get("device_code", "")}',
            {data.get("similarity", "NULL")},
            current_timestamp(),
            current_timestamp()
        )
    """)

    conn.commit()
    cursor.close()
    conn.close()
