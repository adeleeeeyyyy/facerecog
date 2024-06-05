import cv2
import numpy as np

# Inisialisasi webcam
cap = cv2.VideoCapture(0)

while True:
    # Ambil frame dari webcam
    ret, frame = cap.read()
    
    # Ubah warna gambar ke ruang warna HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Definisikan rentang warna kulit dalam HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    # Buat mask untuk menangkap warna kulit dalam rentang yang ditentukan
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Terapkan operasi morfologi untuk membersihkan mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    # Temukan kontur dalam mask
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Gambar kotak di sekitar kontur terbesar (gerakan tangan)
    if len(contours) > 0:
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Tampilkan frame yang telah diolah
    cv2.imshow('Hand Detection', frame)
    
    # Hentikan program jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Tutup webcam dan jendela tampilan
cap.release()
cv2.destroyAllWindows()

