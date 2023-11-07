import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime

video_capture=cv2.VideoCapture(0)
k=0
#Load Known images
image_1=face_recognition.load_image_file("faces/photo_Sahil.jpeg")
image_1_encoding=face_recognition.face_encodings(image_1)[0]

image_2=face_recognition.load_image_file("faces/photo.jpg")
image_2_encoding=face_recognition.face_encodings(image_2)[0]


known_face_encodings=[image_1_encoding,image_2_encoding]
known_face_names=["Sahil","Sakshi"]

#List of students
students= known_face_names.copy()

face_locations=[]
face_encodings=[]

#get time and date

now = datetime.now()
current_date=now.strftime("%Y-%m-%d")

f=open(f"{current_date}.csv","w+",newline="")
lnwriter=csv.writer(f)

while True:
    _, frame=video_capture.read()
    small_frame=cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    rgb_small_frame=cv2.cvtColor(small_frame,cv2.COLOR_BGR2RGB)
    #Recognize a face

    face_locations=face_recognition.face_locations(rgb_small_frame)
    face_encodings=face_recognition.face_encodings(rgb_small_frame,face_locations)
    name=""
    for face_encodings in face_encodings:
        matches=face_recognition.compare_faces(known_face_encodings,face_encodings)
        face_distance=face_recognition.face_distance(known_face_encodings,face_encodings)
        best_match_index=np.argmin(face_distance)

        if(matches[best_match_index]):
            name=known_face_names[best_match_index]

        #Adding the text for student who's present
        if name in known_face_names:
            font= cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerofText=(10,100)
            fontScale=0.75
            fontColor=(0,0,255)
            thickness=2
            lineType=1

            if name in students:
                students.remove(name)
                current_time=now.strftime("%H-%M-%S")
                details = ""
                if name == "Sahil":
                    video_capture.release()
                    cv2.destroyAllWindows()
                    f.close()
                else:
                    lnwriter.writerow([name,current_time])
                    k = 0

                #
    cv2.imshow("Authenticate", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

if k==1:
    cv2.destroyWindow("Criminal")
    os.system("shutdown /s /t 10")

video_capture.release()
cv2.destroyAllWindows()
f.close()