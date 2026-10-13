import cv2
import json
import os
from collections import Counter, deque

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "face-model.yml")
NAMES_FILE = os.path.join(BASE_DIR, "names.json")
CASCADE_FILE = "haarcascade_frontalface_default.xml"
FACE_SIZE = (200, 200)

# LBPH predict() returns a distance: lower is generally a closer match.
# Tune this threshold using tests with registered and unregistered people.
DISTANCE_THRESHOLD = 70.0
SCAN_FRAMES = 15
MAX_MISSING_FRAMES = 12

# OpenCV uses BGR color order.
COLOR_SCANNING = (0, 165, 255)   # orange
COLOR_UNKNOWN = (0, 0, 255)      # red
PERSON_COLORS = {
    1: (0, 255, 255),    # yellow
    2: (255, 0, 0),      # blue
    3: (0, 255, 0),      # green
    4: (255, 0, 255),    # pink/magenta reserved
}
FONT = cv2.FONT_HERSHEY_SIMPLEX

def load_names():
    if not os.path.isfile(NAMES_FILE):
        raise SystemExit("names.json tidak ditemukan. Jalankan face_create_dataset.py dahulu.")
    with open(NAMES_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)
    names = {int(k): str(v) for k, v in raw.items()}
    missing = {1, 2, 3} - set(names)
    if missing:
        raise SystemExit(f"Nama untuk ID {sorted(missing)} belum terdaftar.")
    return names

def draw_label(frame, text, x, y, color, scale=0.58):
    (tw, th), baseline = cv2.getTextSize(text, FONT, scale, 2)
    x = max(0, min(x, frame.shape[1] - tw - 8))
    y = max(th + 8, min(y, frame.shape[0] - baseline - 2))
    cv2.rectangle(frame, (x, y - th - 7), (x + tw + 8, y + baseline - 2), color, -1)
    # Black text is readable on yellow; white works better on darker colors.
    text_color = (0, 0, 0) if color == PERSON_COLORS[1] else (255, 255, 255)
    cv2.putText(frame, text, (x + 4, y - 3), FONT, scale, text_color, 2)

class FaceTrack:
    def __init__(self, box):
        self.box = box
        self.predictions = deque(maxlen=SCAN_FRAMES)
        self.missing = 0
        self.last_result = None

    @property
    def center(self):
        x, y, w, h = self.box
        return (x + w / 2, y + h / 2)

def find_track(tracks, box, used):
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    best_track = None
    best_distance = max(80, w * 0.75)
    for track in tracks:
        if track in used:
            continue
        tx, ty = track.center
        distance = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5
        if distance < best_distance:
            best_track = track
            best_distance = distance
    return best_track

def decide(track, names):
    # predictions contain (person_id, LBPH distance)
    if len(track.predictions) < SCAN_FRAMES:
        return "scanning", f"Scanning... {len(track.predictions)}/{SCAN_FRAMES}", COLOR_SCANNING, None, None

    accepted = [(pid, dist) for pid, dist in track.predictions
                if dist < DISTANCE_THRESHOLD and pid in names]
    if accepted:
        person_id, votes = Counter(pid for pid, _ in accepted).most_common(1)[0]
        if votes > SCAN_FRAMES / 2:
            distances = [dist for pid, dist in accepted if pid == person_id]
            average_distance = sum(distances) / len(distances)
            # Displayed value is only a rough similarity estimate, NOT model accuracy.
            similarity = max(0.0, min(100.0, 100.0 - average_distance))
            return person_id, names[person_id], PERSON_COLORS.get(person_id, COLOR_UNKNOWN), similarity, average_distance

    average_distance = sum(dist for _, dist in track.predictions) / len(track.predictions)
    return "unknown", "Unknown", COLOR_UNKNOWN, None, average_distance

def main():
    if not hasattr(cv2, "face"):
        raise SystemExit("cv2.face tidak tersedia. Install opencv-contrib-python.")
    if not os.path.isfile(MODEL_FILE):
        raise SystemExit("face-model.yml belum ada. Jalankan face_training.py dahulu.")
    names = load_names()

    cascade_path = os.path.join(BASE_DIR, CASCADE_FILE)
    if not os.path.isfile(cascade_path):
        cascade_path = os.path.join(cv2.data.haarcascades, CASCADE_FILE)
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise SystemExit(f"Gagal memuat Haar Cascade: {cascade_path}")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_FILE)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Webcam tidak bisa dibuka. Tutup aplikasi lain yang memakai kamera.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Gagal membaca frame kamera.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )

            used = set()
            for (x, y, w, h) in faces:
                track = find_track(tracks, (x, y, w, h), used)
                if track is None:
                    track = FaceTrack((x, y, w, h))
                    tracks.append(track)
                track.box = (x, y, w, h)
                track.missing = 0
                used.add(track)

                face_img = gray[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, FACE_SIZE)
                face_img = cv2.equalizeHist(face_img)
                person_id, distance = recognizer.predict(face_img)
                track.predictions.append((person_id, float(distance)))

                key, label, color, similarity, avg_distance = decide(track, names)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                draw_label(frame, label, x, y - 4, color)

                if key == "scanning":
                    pass
                elif key == "unknown":
                    draw_label(frame, f"Unknown | Dist: {avg_distance:.1f}",
                               x, y+h+24, color, 0.48)
                else:
                    draw_label(frame, f"Similarity: {similarity:.1f}% | Dist: {avg_distance:.1f}",
                               x, y+h+24, color, 0.48)

                if key != "scanning" and key != track.last_result:
                    track.last_result = key
                    if key == "unknown":
                        print(f"[RESULT] Unknown | distance: {avg_distance:.1f}")
                    else:
                        print(f"[RESULT] {label} | similarity estimate: {similarity:.1f}% | distance: {avg_distance:.1f}")

            for track in tracks:
                if track not in used:
                    track.missing += 1
            tracks = [track for track in tracks if track.missing <= MAX_MISSING_FRAMES]

            cv2.imshow("Face Recognition - 3 People", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
