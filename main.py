import cv2
import FaceDetectionModule

cap = cv2.VideoCapture(0)
face_detector = FaceDetectionModule.FaceDetection(0.75)

while True:
    try:
        success, img = cap.read()
        image_1, position = face_detector.find_faces(img, show_score=True)
        image_2 = face_detector.show_fps()

        cv2.imshow("Image", image_2)
        print(position)
        cv2.waitKey(10)

    except exit(1):
        exit(0)
