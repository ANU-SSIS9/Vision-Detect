import os
import base64
import json
import uuid
from datetime import datetime

import cv2
import numpy as np
import pyodbc
from flask import Flask, jsonify, render_template, request
from ultralytics import YOLO

app = Flask(__name__)

# -----------------------------
# Configuration
# -----------------------------
SQL_SERVER = os.getenv("VISION_SQL_SERVER", r"localhost\SQLEXPRESS")
SQL_DATABASE = os.getenv("VISION_SQL_DATABASE", "VisionDetectDB")
SQL_TRUSTED = os.getenv("VISION_SQL_TRUSTED_CONNECTION", "yes").lower() == "yes"
SQL_USER = os.getenv("VISION_SQL_USER", "")
SQL_PASSWORD = os.getenv("VISION_SQL_PASSWORD", "")

MODEL_NAME = os.getenv("VISION_MODEL", "yolo11n.pt")
CONFIDENCE = float(os.getenv("VISION_CONFIDENCE", "0.35"))

model = None


def get_connection():
    """Create a SQL Server connection using Driver 18 or 17."""
    available = set(pyodbc.drivers())
    if "ODBC Driver 18 for SQL Server" in available:
        driver = "ODBC Driver 18 for SQL Server"
        extra = "TrustServerCertificate=yes;"
    elif "ODBC Driver 17 for SQL Server" in available:
        driver = "ODBC Driver 17 for SQL Server"
        extra = ""
    else:
        raise RuntimeError(
            "Microsoft ODBC Driver 17 or 18 for SQL Server is required."
        )

    if SQL_TRUSTED:
        conn_str = (
            f"DRIVER={{{driver}}};"
            f"SERVER={SQL_SERVER};DATABASE={SQL_DATABASE};"
            "Trusted_Connection=yes;"
            f"{extra}"
        )
    else:
        conn_str = (
            f"DRIVER={{{driver}}};"
            f"SERVER={SQL_SERVER};DATABASE={SQL_DATABASE};"
            f"UID={SQL_USER};PWD={SQL_PASSWORD};"
            f"{extra}"
        )
    return pyodbc.connect(conn_str)


def get_model():
    global model
    if model is None:
        model = YOLO(MODEL_NAME)
    return model


def save_analysis(source_type, source_name, total_objects, counts, average_confidence, original_image):
    session_id = str(uuid.uuid4())
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO DetectionSessions
            (SessionId, SourceType, SourceName, TotalObjects, AverageConfidence, OriginalImage)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            session_id,
            source_type,
            source_name,
            total_objects,
            average_confidence,
            pyodbc.Binary(original_image),
        )

        for class_name, count in counts.items():
            cur.execute(
                """
                INSERT INTO ObjectSummary
                (SessionId, ObjectName, ObjectCount)
                VALUES (?, ?, ?)
                """,
                session_id,
                class_name,
                count,
            )

        conn.commit()
    return session_id


def decode_image(data_url):
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    raw = base64.b64decode(data_url)
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The image could not be decoded.")
    return image


def run_detection(image):
    detector = get_model()
    results = detector.predict(image, conf=CONFIDENCE, verbose=False)

    counts = {}
    confidences = []
    annotated = image.copy()

    for result in results:
        names = result.names
        if result.boxes is None:
            continue

        for box in result.boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            class_name = str(names[class_id])

            counts[class_name] = counts.get(class_name, 0) + 1
            confidences.append(confidence)

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (40, 220, 120), 2)
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(
                annotated,
                label,
                (x1, max(25, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (40, 220, 120),
                2,
            )

    total = sum(counts.values())
    avg = sum(confidences) / len(confidences) if confidences else 0.0

    ok, encoded = cv2.imencode(".jpg", annotated)
    if not ok:
        raise RuntimeError("Could not encode the annotated image.")

    image_base64 = base64.b64encode(encoded.tobytes()).decode("utf-8")
    return counts, total, avg, image_base64


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    try:
        with get_connection() as conn:
            conn.cursor().execute("SELECT 1")
        db = "connected"
    except Exception as exc:
        db = f"error: {exc}"
    return jsonify({"status": "ok", "database": db})


@app.route("/api/analyze", methods=["POST"])
def analyze():
    try:
        payload = request.get_json(silent=True) or {}
        image_data = payload.get("image")
        source_type = payload.get("sourceType", "Webcam")
        source_name = payload.get("sourceName", "Browser Camera")

        if not image_data:
            return jsonify({"error": "No image was supplied."}), 400

        image = decode_image(image_data)
        counts, total, avg, annotated = run_detection(image)

        ok, original_encoded = cv2.imencode(".jpg", image)
        if not ok:
            raise RuntimeError("Could not encode the submitted image.")
        original_image = original_encoded.tobytes()

        session_id = save_analysis(
            source_type, source_name, total, counts, avg, original_image
        )

        return jsonify(
            {
                "sessionId": session_id,
                "totalObjects": total,
                "averageConfidence": round(avg * 100, 2),
                "counts": counts,
                "annotatedImage": f"data:image/jpeg;base64,{annotated}",
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/history")
def history():
    try:
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT TOP 50
                    SessionId, SourceType, SourceName, TotalObjects,
                    AverageConfidence, CreatedAt
                FROM DetectionSessions
                ORDER BY CreatedAt DESC
                """
            )
            rows = cur.fetchall()

        result = []
        for row in rows:
            result.append(
                {
                    "sessionId": str(row.SessionId),
                    "sourceType": row.SourceType,
                    "sourceName": row.SourceName,
                    "totalObjects": int(row.TotalObjects),
                    "averageConfidence": round(float(row.AverageConfidence) * 100, 2),
                    "createdAt": row.CreatedAt.strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/dashboard")
def dashboard():
    try:
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT
                    COUNT(*) AS TotalSessions,
                    ISNULL(SUM(TotalObjects), 0) AS TotalObjects,
                    ISNULL(AVG(AverageConfidence), 0) AS AvgConfidence
                FROM DetectionSessions
                """
            )
            stats = cur.fetchone()

            cur.execute(
                """
                SELECT TOP 10 ObjectName, SUM(ObjectCount) AS TotalCount
                FROM ObjectSummary
                GROUP BY ObjectName
                ORDER BY TotalCount DESC, ObjectName
                """
            )
            objects = [
                {"name": r.ObjectName, "count": int(r.TotalCount)}
                for r in cur.fetchall()
            ]

        return jsonify(
            {
                "sessions": int(stats.TotalSessions),
                "objects": int(stats.TotalObjects),
                "confidence": round(float(stats.AvgConfidence) * 100, 2),
                "topObjects": objects,
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
