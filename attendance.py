import cv2
from face_recognition.api import face_encodings
import numpy as np
import face_recognition
import os
from datetime import datetime




path ='images/'
images = []
personName = []
my_list = os.listdir(path)
for i in my_list:
    current_img = cv2.imread(os.path.join(path,i))
    images.append(current_img)
  
    personName.append(i.split('.')[0])
print(personName)




# face encoding
# dlib finds 128 unique points from a face (HOG)
def face_encoder(images):
    encode_list = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]  # 0th element
        
        encode_list.append(encode)
    return encode_list



encodings = face_encoder(images)

print('encodings generated!!')

def attendance(name):
    with open('attendance.csv',"r+") as f:
        my_data = f.readlines()
        name_list = []
        for line in my_data:
            entry = line.split(',')
            name_list.append(entry[0])
        if name not in name_list:
            time_now = datetime.now()
            time = time_now.strftime('%H:%M:%S')
            date = time_now.strftime('%d/%m/%Y')

            f.writelines(f'{name},{time},{date}\n')


# camera

# cap = cv2.VideoCapture(0)
# address = 'https://192.168.43.1:8080/video'
# cap.open(address)

cap = cv2.VideoCapture(0)


img_counter = 0



while True:
    ret, frame = cap.read()
    if not ret:
        print("failed to grab frame")
        break
    # cv2.imshow("Attendance", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1
    #ret, frame = cap.read()

    # resize taakee sab same format

    faces = cv2.resize(frame,(0,0),None, 0.25, 0.25)

    faces = cv2.cvtColor(faces,cv2.COLOR_BGR2RGB)

    # find face location in camera

    locations = face_recognition.face_locations(faces)

    # face encodings

    currrent_encodings = face_recognition.face_encodings(faces, locations)

    # is the face matching ?  and face distance

    for encodeFace, faceLocation in zip(currrent_encodings,locations):
        matches = face_recognition.compare_faces(encodings, encodeFace)
        #print(matches)
         # euclidean distance (how similar the face is )
        face_distance = face_recognition.face_distance(encodings,encodeFace)
        #print(face_distance)
        match_index = np.argmin(face_distance)

        if matches[match_index]:
            name = personName[match_index].upper()
            #print(name)

            y1, x2, y2, x1 = faceLocation

            # now multiply the parameters by 4 cause uppar humne resize kya tha
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            
            cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0),3)
            cv2.rectangle(frame, (x1, y2-35), (x2,y2), (0,255,255), cv2.FILLED)
    
            cv2.putText(frame,name, (x1+6, y2+6), cv2.FONT_HERSHEY_COMPLEX,1, (0,0,255),2)
    
            attendance(name)
    cv2.imshow('Attendance',frame)


    if cv2.waitKey(10) == 13:
        break
cap.release()
cv2.destroyAllWindows()














