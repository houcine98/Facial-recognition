from django.shortcuts import render
from django.http import HttpResponse
from django.shortcuts import redirect
from django.urls import reverse
from .models import Patient
import cv2
import time
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import numpy as np
import requests


new_model=tf.keras.models.load_model('Fer_2013/fmodel.h5')
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
# Create your views here.

def session (request):
    return render(request,"session/index.html")

def patient_details (request):
    if request.method == 'POST': 
        name = request.POST.get('name')
        age = request.POST.get('age')
        health = request.POST.get('health')
        disease = request.POST.get('disease')
        

        timestamp = int(time.time()) 
        video_filename = f'Fer_2013/videos/output_{timestamp}.avi'


         

        # Constants
        path = "haarcascade_frontalface_default.xml"
        font_scale = 1.5
        font = cv2.FONT_HERSHEY_PLAIN
        rectangle_bgr = (255, 255, 255)

        # Create a black image
        img = np.zeros((500, 500))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(video_filename, fourcc, 20.0, (640, 480))  # Adjust resolution as needed
        # Set text and calculate text box size
        text = "same text in a box"
        (text_width, text_height), _ = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)
        text_offset_x = 10
        text_offset_y = img.shape[0] - 25
        box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height - 2))

        # Draw text box and text
        cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
        cv2.putText(img, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(0, 0, 0), thickness=1)

        cap = cv2.VideoCapture(1)  # Try to open webcam 1
        if not cap.isOpened():  # If webcam 1 is not available, try webcam 0
            cap = cv2.VideoCapture(0)
        if not cap.isOpened():  # If no webcam is available, raise an error
            raise IOError("Cannot open webcam")

        # Load face cascade classifier
        faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, 1.1, 4)

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = frame[y:y + h, x:x + w]

                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                facess = faceCascade.detectMultiScale(roi_gray)

                if len(facess) == 0:
                    print("face not detected")
                else:
                    for (ex, ey, ew, eh) in facess:
                        face_roi = roi_color[ey: ey + eh, ex: ex + ew]

            if 'face_roi' in locals():  # If face_roi is defined
                final_image = cv2.resize(face_roi, (224, 224))
                final_image = np.expand_dims(final_image, axis=0)
                final_image = final_image / 255.0

                Predictions = new_model.predict(final_image)

                emotions = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
                status = emotions[np.argmax(Predictions)]

                x1, y1, w1, h1 = 0, 0, 175, 75
                cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0, 0, 0), -1)
                cv2.putText(frame, status, (x1 + int(w1 / 10), y1 + int(h1 / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, status, (100, 150), font, 3, (0, 0, 255), 2, cv2.LINE_4)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255))

            cv2.imshow('Face Emotion Detection', frame)
            out.write(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        patient = Patient(name=name, age=age, health=health, disease=disease, video_file=video_filename)
        patient.save()
        return redirect('record')


    return render(request,"session/patient_details.html")


def record (request):
    all_patients = Patient.objects.all()

    context = {'all_patients': all_patients}


    return render(request,"session/record.html",context)


