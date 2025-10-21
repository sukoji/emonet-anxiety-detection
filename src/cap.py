import os
import queue
import threading
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = ROOT_DIR / "assets"
MODEL_PATH = ASSETS_DIR / "models" / "fer_student_kd.onnx"
LABELS_PATH = ASSETS_DIR / "data" / "anxiety_classifier_labels.csv"
CHART_BG_PATH = ASSETS_DIR / "ui" / "valence_arousal_chart_bg.png"


def main() -> None:
    if not LABELS_PATH.exists():
        raise FileNotFoundError(f"Missing training data: {LABELS_PATH}. See assets/README.md")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing ONNX model: {MODEL_PATH}. See assets/README.md")

    lda, svm_model_lda, scaler, min_score, max_score = train_model(
        pd.read_csv(LABELS_PATH, encoding="ISO-8859-1")
    )

    frame_queue = queue.Queue(maxsize=16)
    result_queue = queue.Queue(maxsize=8)

    webcam_thread = threading.Thread(
        target=read_webcam_frames,
        args=(frame_queue, result_queue, 0),
        daemon=True,
    )
    batch_thread = threading.Thread(
        target=process_batches,
        args=(frame_queue, result_queue, lda, svm_model_lda, scaler, min_score, max_score),
        daemon=True,
    )

    torch.backends.cudnn.benchmark = False
    webcam_thread.start()
    batch_thread.start()
    webcam_thread.join()
    batch_thread.join()


def draw_arousal_valence_on_face(frame, x, y, w, h, valence, arousal, last_anxiety, last_score):
    if isinstance(valence, np.ndarray):
        valence = valence.item()
    if isinstance(arousal, np.ndarray):
        arousal = arousal.item()

    graph_width, graph_height = 250, 220
    graph_x, graph_y = 10, frame.shape[0] - graph_height - 10

    if CHART_BG_PATH.exists():
        graph_bg = cv2.imread(str(CHART_BG_PATH))
        graph_bg = cv2.resize(graph_bg, (graph_width, graph_height))
        frame[graph_y : graph_y + graph_height, graph_x : graph_x + graph_width] = graph_bg

    center_x = int(graph_x + graph_width / 2 + (valence * (graph_width / 2)))
    center_y = int(graph_y + graph_height / 2 - (arousal * (graph_height / 2)))
    cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)

    valence_normalized = (valence + 1) / 2
    arousal_normalized = (arousal + 1) / 2
    bar_width = 10
    valence_bar_height = int(valence_normalized * h)

    color = (0, 255, 0) if last_anxiety == 0 else (0, 0, 255)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    score_text = f"Score: {last_score:.2f}" if last_score is not None else "Score: N/A"
    cv2.putText(
        frame,
        f'Anxiety: {"Yes" if last_anxiety == 1 else "No"}, {score_text}',
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2,
    )

    cv2.rectangle(
        frame,
        (x - bar_width - 10, y + h - valence_bar_height),
        (x - 10, y + h),
        color,
        -1,
    )
    cv2.rectangle(
        frame,
        (x, y + h + 10),
        (x + int(w * arousal_normalized), y + h + 10 + bar_width),
        color,
        -1,
    )
    cv2.putText(frame, f"Valence: {valence:.2f}", (x, y + h + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Arousal: {arousal:.2f}", (x, y + h + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def read_webcam_frames(frame_queue, result_queue, input_index):
    cap = cv2.VideoCapture(input_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    last_anxiety = None
    last_score = None
    last_valence = None
    last_arousal = None
    last_bbox = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            if not result_queue.empty():
                x, y, w, h, last_anxiety, last_score, last_valence, last_arousal = result_queue.get()
                last_bbox = (x, y, w, h)
                if isinstance(last_score, np.ndarray):
                    last_score = last_score.item()

            if last_valence is not None and last_arousal is not None and last_bbox is not None:
                x, y, w, h = last_bbox
                draw_arousal_valence_on_face(
                    frame, x, y, w, h, last_valence, last_arousal, last_anxiety, last_score
                )

            if frame_queue.full():
                frame_queue.get()
            frame_queue.put(frame)
            cv2.imshow("Face Detection and Emotion Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def process_batches(frame_queue, result_queue, lda, svm_model_lda, scaler, min_score, max_score):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    onnx_model = onnx.load(str(MODEL_PATH))
    onnx.checker.check_model(onnx_model)
    ort_session = ort.InferenceSession(str(MODEL_PATH))

    frames = []
    while True:
        if not frame_queue.empty():
            frames.append(frame_queue.get())
            if len(frames) == 1:
                process_batch(
                    face_cascade,
                    frames,
                    result_queue,
                    lda,
                    svm_model_lda,
                    scaler,
                    min_score,
                    max_score,
                    ort_session,
                )
                frames = []


def process_batch(
    face_cascade,
    frames,
    result_queue,
    lda,
    svm_model_lda,
    scaler,
    min_score,
    max_score,
    ort_session,
):
    for frame in frames:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_roi = frame[y : y + h, x : x + w]
            input_data = cv2.resize(face_roi, (256, 256))
            input_data = np.expand_dims(input_data, axis=0).astype("float32") / 255.0

            input_name = ort_session.get_inputs()[0].name
            output_names = [output.name for output in ort_session.get_outputs()]
            outputs = ort_session.run(output_names, {input_name: input_data})

            arousal, valence = outputs[0][0], outputs[2][0]
            anxiety, score = predict_anxiety(lda, svm_model_lda, scaler, min_score, max_score, arousal, valence)
            result_queue.put((x, y, w, h, anxiety, score, valence, arousal))


def train_model(df):
    imputer = SimpleImputer(strategy="mean")
    df[["arousal", "valence"]] = imputer.fit_transform(df[["arousal", "valence"]])

    x = df[["arousal", "valence"]]
    y = df["emotion"]
    y_binary = np.where((y == 0) | (y == 5), 0, 1)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_binary, test_size=0.2, random_state=42)

    lda = LDA()
    x_train_lda = lda.fit_transform(x_train, y_train)
    x_test_lda = lda.transform(x_test)

    min_score = x_test_lda.min()
    max_score = x_test_lda.max()

    svm_model_lda = SVC(kernel="rbf", C=1, gamma="scale")
    svm_model_lda.fit(x_train_lda, y_train)
    return lda, svm_model_lda, scaler, min_score, max_score


def predict_anxiety(lda, svm_model_lda, scaler, min_score, max_score, arousal, valence):
    new_data = pd.DataFrame([[arousal, valence]], columns=["arousal", "valence"])
    new_data_scaled = scaler.transform(new_data)
    new_data_lda = lda.transform(new_data_scaled)

    score = ((new_data_lda - min_score) / (max_score - min_score)) * 100
    anxiety_pred = svm_model_lda.predict(new_data_lda)
    return anxiety_pred[0], score[0]


if __name__ == "__main__":
    main()
