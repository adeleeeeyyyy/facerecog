import cv2

# Inisialisasi cascade classifier untuk deteksi wajah
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Fungsi untuk mendeteksi wajah pada gambar
def detect_faces(img):
    # Konversi gambar menjadi grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Deteksi wajah pada gambar
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Gambar kotak di sekitar wajah yang terdeteksi
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return img

# Inisialisasi webcam
cap = cv2.VideoCapture(0)

while True:
    # Ambil frame dari webcam
    ret, frame = cap.read()
    # Deteksi wajah pada frame
    frame = detect_faces(frame)
    # Tampilkan frame yang telah diolah
    cv2.imshow('Face Detection', frame)
    # Hentikan program jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Tutup webcam dan jendela tampilan
cap.release()
cv2.destroyAllWindows()

