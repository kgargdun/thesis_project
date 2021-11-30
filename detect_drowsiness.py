from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import argparse
import imutils
import time
import dlib
import cv2
import playsound
from twilio.rest import Client

import geocoder

g = geocoder.ip('me')
print(str(g.latlng))

account_sid = 'XXX'
auth_token = 'XXX'
client = Client(account_sid, auth_token)


def eye_aspect_ratio(captured_eye):
    a = dist.euclidean(captured_eye[1], captured_eye[5])
    b = dist.euclidean(captured_eye[2], captured_eye[4])
    c = dist.euclidean(captured_eye[0], captured_eye[3])

    calc_ear = (a + b) / (2.0 * c)
    return calc_ear


def calculate_ear(outline):
    (l_start, l_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (r_start, r_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    left_eye = outline[l_start:l_end]
    right_eye = outline[r_start:r_end]

    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)

    calc_ear = (left_ear + right_ear) / 2.0
    return calc_ear, left_eye, right_eye


def ring_bell():
    playsound.playsound('alarm.wav', False)


ap = argparse.ArgumentParser()
ap.add_argument("-a", "--alarm", type=str, default="", help="path alarm .WAV file")
ap.add_argument("-w", "--webcam", type=int, default=0, help="index of webcam on system")
args = vars(ap.parse_args())

EAR_THRESH_RATIO = 0.25
ADJ_FRAMES = 25
message_eye = 1

COUNTER = 0
ALARM_ON = False

print("-> Starting up...")
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

print("-> Starting Video Stream")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)

while True:

    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rectangle = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                          flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in rectangle:
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        eye = calculate_ear(shape)
        ear = eye[0]
        leftEye = eye[1]
        rightEye = eye[2]

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        lip = shape[48:60]
        cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

        if ear < EAR_THRESH_RATIO:
            COUNTER += 1

            if COUNTER >= ADJ_FRAMES:
                if not ALARM_ON:
                    ALARM_ON = True
                    if message_eye == 1:
                        message = client.messages \
                            .create(
                            body="Drowsy driver detected at location " + str(g.latlng),
                            from_='XXX',
                            to='XXX'
                        )
                        print(message.sid)
                        message_eye = 0
                        print('SMS Sent for drowsy')

                    if args["alarm"] != "":
                        t = Thread(target=ring_bell, args=(args["alarm"],))
                        t.deamon = True
                        t.start()

                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        else:
            COUNTER = 0
            message_eye = 1
            ALARM_ON = False

        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
