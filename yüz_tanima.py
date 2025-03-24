import cv2
import numpy as np
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk

AGE_MODEL = "models/age_net.caffemodel"
AGE_PROTO = "models/age_deploy.prototxt"
GENDER_MODEL = "models/gender_net.caffemodel"
GENDER_PROTO = "models/gender_deploy.prototxt"
FACE_PROTO = "models/opencv_face_detector.pbtxt"
FACE_MODEL = "models/opencv_face_detector_uint8.pb"

try:
    age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)
    gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTO, GENDER_MODEL)
    face_net = cv2.dnn.readNetFromTensorflow(FACE_MODEL, FACE_PROTO)
    print("Modeller başarıyla yüklendi.")
except Exception as e:
    print(f"Model yükleme hatası: {e}")
    exit()

AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]
GENDER_BUCKETS = ["Erkek", "Kadın"]

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Kamera açılamadı.")
    exit()

root = tk.Tk()
root.title("Yüz Tanıma Uygulaması")
root.configure(bg="black")
root.geometry("800x600")

title_label = tk.Label(root, text="YÜZ TANIMA UYGULAMASI", font=("Verdana", 18, "bold"), fg="white", bg="black")
title_label.pack(pady=10)

video_label = Label(root)
video_label.pack()

info_label = tk.Label(root, text="Lütfen yüzünüzü ortalayınız", font=("Verdana", 12, "bold"), fg="red", bg="black")
info_label.pack(pady=10)

result_label = Label(root, text="Kamera Açılıyor...", font=("Verdana", 14), fg="white", bg="black")
result_label.pack(pady=10)

capture_button = Button(root, text="Fotoğraf Çek", font=("Arial", 14), bg="gray", fg="white", command=lambda: capture_image())
capture_button.pack(pady=10)

previous_gender = None

def update_frame():
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1)

        # Yüz tespiti
        face_blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], swapRB=False)
        face_net.setInput(face_blob)
        detections = face_net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (x, y, x2, y2) = box.astype("int")
                cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)

        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

    video_label.after(10, update_frame)


def capture_image():
    ret, frame = cap.read()
    last_predictions = []  # Tahminleri saklamak için liste
    if not ret:
        return

    face_blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], swapRB=False)
    face_net.setInput(face_blob)
    detections = face_net.forward()
    prediction_text = "Yüz Bulunamadı"

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Yüzü bulma eşik değeri
            x, y, x2, y2 = (detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])).astype("int")
            face = frame[y:y2, x:x2]

            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), [78.42, 87.76, 114.89], swapRB=False)

            age_net.setInput(blob)
            age = AGE_BUCKETS[age_net.forward()[0].argmax()]

            gender_net.setInput(blob)
            gender = GENDER_BUCKETS[gender_net.forward()[0].argmax()]

            global previous_gender
            if previous_gender is None:
                previous_gender = gender
            else:
                gender = previous_gender

            prediction_text = f"{gender}, {age}"

    result_label.config(text=f"Tahmin: {prediction_text}", fg="green")
    capture_button.config(text="Tekrar Çek", bg="red")


update_frame()
root.mainloop()
cap.release()
cv2.destroyAllWindows()
