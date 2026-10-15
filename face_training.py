import cv2
import numpy as np
import os
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_FILE = os.path.join(BASE_DIR, "face-model.yml")
FACE_SIZE = (200, 200)
VALID_IDS = {1, 2, 3}

def main():
    if not os.path.isdir(DATASET_DIR):
        raise SystemExit("Folder dataset belum ada. Jalankan face_create_dataset.py dahulu.")

    faces = []
    ids = []
    for filename in sorted(os.listdir(DATASET_DIR)):
        stem, ext = os.path.splitext(filename)
        if ext.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        parts = stem.split("-")
        if len(parts) != 3 or parts[0] != "Person" or not parts[1].isdigit():
            print(f"Lewati nama file yang tidak sesuai format: {filename}")
            continue
        person_id = int(parts[1])
        if person_id not in VALID_IDS:
            print(f"Lewati ID di luar 1-3: {filename}")
            continue

        path = os.path.join(DATASET_DIR, filename)
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None or image.size == 0:
            print(f"Gagal membaca: {filename}")
            continue
        image = cv2.resize(image, FACE_SIZE)
        image = cv2.equalizeHist(image)
        faces.append(image)
        ids.append(person_id)

    if not faces:
        raise SystemExit("Tidak ada gambar dataset yang valid.")
    counts = Counter(ids)
    print("Jumlah sampel per ID:", dict(sorted(counts.items())))
    missing_ids = VALID_IDS - set(ids)
    if missing_ids:
        raise SystemExit(
            f"Dataset untuk ID {sorted(missing_ids)} belum ada. "
            "Daftarkan ketiga orang sebelum training."
        )

    print(f"Training LBPH dengan {len(faces)} gambar...")
    recognizer = cv2.face.LBPHFaceRecognizer_create(
    radius=2,
    neighbors=8,
    grid_x=8,
    grid_y=8
    )
    recognizer.train(faces, np.array(ids, dtype=np.int32))
    recognizer.write(MODEL_FILE)
    print("Training selesai.")
    print(f"Model disimpan ke: {MODEL_FILE}")

if __name__ == "__main__":
    main()
