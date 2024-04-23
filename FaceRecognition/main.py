import cv2
import threading

from deepface import DeepFace

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)



reference_image = cv2.imread("FaceReference.jpg")


def check_face(frame):
    global face_match
    try:
        if DeepFace.verify(frame,reference_image.copy())['verified']:
            face_match = True
        else:
            face_match = False

    except ValueError:
        face_match = False




def destroy_windows():
    cv2.destroyAllWindows()

def main():
    counter = 0
    face_match = False
    while True:
        return_value, frame = cap.read()

        if return_value:
            if counter % 30 == 0:
                try:
                    threading.Thread(target=check_face, args=(frame.copy(),)).start()

                except ValueError:
                    pass
            counter += 1
            if face_match:
                cv2.putText(frame, "MATCH", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (49, 205, 49), 3)
            else:
                cv2.putText(frame, "No MATCH", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (22, 0, 201), 3)

            cv2.imshow("Video", frame)

        key = cv2.waitKey(1)
        if key == ord("q"):
             break