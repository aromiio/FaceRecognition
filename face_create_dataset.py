import cv2
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
NAMES_FILE = os.path.join(BASE_DIR, "names.json")
CASCADE_FILE = "haarcascade_frontalface_default.xml"

SAMPLES_PER_PERSON = 100
CAPTURE_EVERY_N_FRAMES = 5
FACE_SIZE = (200, 200)

# OpenCV colors use BGR order.
COLOR_SCANNING = (0, 165, 255)  # orange
PERSON_COLORS = {
    1: (0, 255, 255),    # yellow
    2: (255, 0, 0),      # blue
    3: (0, 255, 0),      # green
    4: (255, 0, 255),    # pink/magenta, reserved
}

def load_names():
    if not os.path.isfile(NAMES_FILE):
        return {}
    try:
        with open(NAMES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {int(k): str(v) for k, v in data.items()}
    except (json.JSONDecodeError, ValueError):
        raise SystemExit("names.json rusak. Periksa format file tersebut.")

def save_names(names):
    with open(NAMES_FILE, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in sorted(names.items())},
                  f, indent=2, ensure_ascii=False)

def main():
    try:
        person_id = int(input("Masukkan ID orang (1, 2, 3): ").strip())
    except ValueError:
        raise SystemExit("ID harus berupa angka 1, 2, atau 3.")

    names = load_names()
    previous_name = names.get(person_id, "")
    prompt = f"Masukkan nama orang ID {person_id}"
    prompt += ": "
    entered_name = input(prompt).strip()
    name = entered_name 
    if not name:
        raise SystemExit("Nama tidak boleh kosong.")
    names[person_id] = name
    save_names(names)

    cascade_path = os.path.join(BASE_DIR, CASCADE_FILE)
    if not os.path.isfile(cascade_path):
        cascade_path = os.path.join(cv2.data.haarcascades, CASCADE_FILE)
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise SystemExit(f"Gagal memuat Haar Cascade: {cascade_path}")

    os.makedirs(DATASET_DIR, exist_ok=True)
    prefix = f"Person-{person_id}-"
    existing_numbers = []
    for filename in os.listdir(DATASET_DIR):
        stem, ext = os.path.splitext(filename)
        if ext.lower() in (".jpg", ".jpeg", ".png") and stem.startswith(prefix):
            suffix = stem[len(prefix):]
            if suffix.isdigit():
                existing_numbers.append(int(suffix))
    start_number = max(existing_numbers, default=0)

    # DirectShow helps avoid webcam initialization issues on Windows.
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Webcam tidak bisa dibuka. Tutup aplikasi lain yang memakai kamera.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    color = PERSON_COLORS[person_id]
    saved = 0
    frame_number = 0
    print(f"\nMengambil {SAMPLES_PER_PERSON} sampel untuk {name} (ID {person_id}).")
    print("Hadap kamera, lalu ubah posisi kepala sedikit. Tekan q untuk berhenti.\n")

    try:
        while saved < SAMPLES_PER_PERSON:
            ok, frame = cap.read()
            if not ok:
                print("Frame kamera gagal dibaca.")
                break

            frame_number += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )

            if len(faces) == 1:
                x, y, w, h = faces[0]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"SCANNING: {name}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLOR_SCANNING, 2)

                if frame_number % CAPTURE_EVERY_N_FRAMES == 0:
                    face_img = gray[y:y+h, x:x+w]
                    face_img = cv2.resize(face_img, FACE_SIZE)
                    face_img = cv2.equalizeHist(face_img)
                    filename = f"{prefix}{start_number + saved + 1}.jpg"
                    filepath = os.path.join(DATASET_DIR, filename)
                    if cv2.imwrite(filepath, face_img):
                        saved += 1
                        print(f"Tersimpan {saved}/{SAMPLES_PER_PERSON}: {filename}")
            elif len(faces) > 1:
                cv2.putText(frame, "Pastikan hanya satu wajah terlihat", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Posisikan wajah di depan kamera", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_SCANNING, 2)

            cv2.putText(frame, f"ID {person_id} - {name}: {saved}/{SAMPLES_PER_PERSON}",
                        (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.imshow("Create Face Dataset", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Pengambilan dihentikan. Jalankan ulang untuk melanjutkan sampel.")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"\nSelesai. Sampel baru: {saved}. Folder dataset: {DATASET_DIR}")

if __name__ == "__main__":
    main()
