import cv2
from datetime import datetime
import webbrowser as w
import os
import time
import keyboard

cascade_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cascade_eye = cv2.CascadeClassifier('haarcascade_eye.xml')
cascade_smile = cv2.CascadeClassifier('haarcascade_smile.xml')
global i
i = []
def time_counter():
    i.append(0)
def time_counter_reset():
    i.clear()

def detection(grayscale, img):
    face = cascade_face.detectMultiScale(grayscale, 3, 5)
    for (x_face, y_face, w_face, h_face) in face:
        cv2.rectangle(img, (x_face, y_face), (x_face+w_face, y_face+h_face), (255, 130, 0), 2)
        ri_grayscale = grayscale[y_face:y_face+h_face, x_face:x_face+w_face]
        ri_color = img[y_face:y_face+h_face, x_face:x_face+w_face]

        eye = cascade_eye.detectMultiScale(ri_grayscale, 1.2, 18)
        if len(eye) != 0:
            for (x_eye, y_eye, w_eye, h_eye) in eye:
                cv2.rectangle(ri_color,(x_eye, y_eye),(x_eye+w_eye, y_eye+h_eye), (0, 180, 60), 2)
            time_counter_reset()

        else:
            start_time = datetime.now()
            #print("Глаза закрыты")
            time_counter()




        # smile = cascade_smile.detectMultiScale(ri_grayscale, 1.6, 4)
        # for (x_smile, y_smile, w_smile, h_smile) in smile:
        #     cv2.rectangle(ri_color,(x_smile, y_smile),(x_smile+w_smile, y_smile+h_smile), (255, 0, 130), 2)
    return img


vc = cv2.VideoCapture(0)

while True:
    print(len(i))
    _, img = vc.read()
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    final = detection(grayscale, img)
    if len(i) == 150:
        w.open('https://www.youtube.com/watch?v=-ni_Gn1IVWY')
    else:
        pass
    try:
        if keyboard.is_pressed('Space'):
            os.system("taskkill /im chrome.exe")
    except:
        pass
    # print(datetime.now() - start_time)
    # start_time = 0
    cv2.imshow('Video', final)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vc.release()
cv2.destroyAllWindows()




# img = cv2.imread('photos/face1.png')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# face = cv2.CascadeClassifier('face.xml')
# results = face.detectMultiScale(img, scaleFactor=1.1, minNeighbors=3)
#
# for (x, y, w, h) in results:
#     print((x, y, w, h) )
#
# for (x, y, w, h) in results:
#     cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), thickness=3)
# cv2.imshow('Result', img)
# cv2.waitKey(0)



# cap = cv2.VideoCapture(0)
# cap.set(3, 1600)
# cap.set(4,900)
#
# #face = cv2.CascadeClassifier('face_smile.xml')
# face = cv2.CascadeClassifier(cv2.data.haarcascades +'face_smile.xml')
#
#
# while True:
#     success, img = cap.read()
#     #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     #img = cv2.Canny(img, 255, 255, 255)
#     results = face.detectMultiScale(img, 2.5, 10)
#     #img = cv2.flip(img, 1)
#     for (x, y, w, h) in results:
#         print(results)
#         cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), thickness=3)
#
#     cv2.imshow('result', img)
#     cv2.waitKey(50)



# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_eye.xml')
# smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_smile.xml')
#
# faces  = face_cascade.detectMultiScale(gray, 1.3, 5)

# def detect(gray, frame):
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x, y), ((x + w), (y + h)), (255, 0, 0), 2)
#         roi_gray = gray[y:y + h, x:x + w]
#         roi_color = frame[y:y + h, x:x + w]
#         smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
#
#         for (sx, sy, sw, sh) in smiles:
#             cv2.rectangle(roi_color, (sx, sy), ((sx + sw), (sy + sh)), (0, 0, 255), 2)
#     return frame
#
# video_capture = cv2.VideoCapture(0)
# while video_capture.isOpened():
#     # Captures video_capture frame by frame
#     _, frame = video_capture.read()
#
#     # To capture image in monochrome
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     # calls the detect() function
#     canvas = detect(gray, frame)
#
#     # Displays the result on camera feed
#     cv2.imshow('Video', canvas)
#
#     # The control breaks once q key is pressed
#     if cv2.waitKey(1) & 0xff == ord('q'):
#         break
#
# # Release the capture once all the processing is done.
# video_capture.release()
# cv2.destroyAllWindows()











# import numpy as np
#
# photo = np.zeros((450, 450, 3), dtype='uint8')
# # photo[100:150, 100:150] = 143, 8, 156
# # cv2.rectangle(photo, (0, 0), (100, 100), (143, 8, 130), thickness=3)
# # cv2.line(photo, (0, 0), (100, 100), (143, 8, 130), thickness=3)
#
# cv2.putText(photo, 'DA', (100,150), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0), thickness=1)
#
# cv2.imshow('Result', photo)
# cv2.waitKey(0)





# import cv2
# import numpy as np
#
# img = cv2.imread("photos/1.jpg")
# new_img = cv2.resize(img, (img.shape[1]*2, img.shape[0]*2))
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = cv2.Canny(img, 500, 500)
#
# kernel = np.ones((5, 5), np.uint8)
# img = cv2.dilate(img, kernel, iteration=1)
#
# print(img.shape)
# # print(new_img.shape)
# #cv2.imshow('Result', img[0:100, 0:150])
#
#
# cv2.imshow('Result', img)
#
# cv2.waitKey(0)
#
#
# # import cv2
# #
# # cap = cv2.VideoCapture(0)
# # cap.set(3, 1600)
# # cap.set(4,900)
# # while True:
# #     success, img = cap.read()
# #     cv2.imshow('result', img)
# #     cv2.waitKey(50)
