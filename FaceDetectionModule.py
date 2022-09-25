import cv2
import mediapipe
import time


class FaceDetection:
    def __init__(self, min_detection_confidence=0.5, model_selection=0):
        self.min_det = min_detection_confidence
        self.model_sel = model_selection

        self.mpFaceDetection = mediapipe.solutions.face_detection
        self.mpDraw = mediapipe.solutions.drawing_utils
        self.FaceDetection = self.mpFaceDetection.FaceDetection(0.75)

        self.image = None
        self.cTime = time.time()
        self.pTime = 1
        self.b_boxs = list()

    def find_faces(self, image, show_score=True, show_rec=True):
        img = image
        self.image = img

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.FaceDetection.process(img_rgb)

        if results.detections:
            self.b_boxs = list()
            for id_, detection in enumerate(results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

                if show_rec:
                    self.fancy_draw(img, bbox)

                self.b_boxs.append([id_, bbox, detection.score])

                if show_score:
                    cv2.putText(img, f'{round(float(detection.score[0]) * 100, 2)} %',
                                (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
        return img, self.b_boxs

    def show_fps(self):
        img = self.image

        self.cTime = time.time()
        fps = 1 / (self.cTime - self.pTime)
        self.pTime = self.cTime

        cv2.putText(img, f'FPS: {int(fps)}', (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

        return img

    @staticmethod
    def fancy_draw(img, bbox, le=30, t=3):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h
        cv2.rectangle(img, bbox, (0, 255, 0), 1)

        cv2.line(img, (x, y), (x+le, y), (0, 255, 0), t)
        cv2.line(img, (x, y), (x, y+le), (0, 255, 0), t)

        cv2.line(img, (x1, y), (x1 - le, y), (0, 255, 0), t)
        cv2.line(img, (x1, y), (x1, y + le), (0, 255, 0), t)

        cv2.line(img, (x, y1), (x + le, y1), (0, 255, 0), t)
        cv2.line(img, (x, y1), (x, y1 - le), (0, 255, 0), t)

        cv2.line(img, (x1, y1), (x1 - le, y1), (0, 255, 0), t)
        cv2.line(img, (x1, y1), (x1, y1 - le), (0, 255, 0), t)

        return img
