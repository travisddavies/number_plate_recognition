from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import easyocr
import numpy as np


class NumberPlateRecogniser:
    """
    A number plate recognition model that both detects number plates
    and extracts the text from the detected number plates.

    For initialisation, choose either size='n' or size='s'.
    """

    def __init__(self, size='n'):
        assert size in {'n', 's'}, 'size of NumberPlateRecogniser must be \
                s or n'

        # Number plate detection model.
        self._detector = YOLO(f'best_weights/yolov8{size}')
        # OCR model for number plate text extraction.
        self._recogniser = easyocr.Reader(['en'])

    # Detects the number plate in an image.
    def detect(self, image):
        return self._detector(image)

    def recognise(self, cropped_image):
        """
        Recognises the number plate text of a cropped_image.

        input: cropped_image - cropped to number plate

        returns: the text of the number plate as a list
        """
        w = cropped_image.shape[1]
        min_text_size = int(0.25 * w)

        return self._recogniser.readtext(cropped_image,
                                         allowlist=self.allowable_chars,
                                         min_size=min_text_size,
                                         detail=0,
                                         paragraph=True,
                                         width_ths=w)

    def extract_bboxes(self, image):
        """
        Extracts the bboxes of the given image that are predicted to be number
        plates.

        input: the image to extract text from
        returns: list of tuples representing bboxe coordinates x1, y1, x2, y2
        """
        predictions = self.detect(image)
        extracted_bboxes = []

        for prediction in predictions:
            boxes = prediction.boxes()
            for box in boxes:
                xyxy = box.xyxy.squeeze()
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), \
                    int(xyxy[3])
                extracted_bboxes.append((x1, y1, x2, y2))

        return extracted_bboxes

    def extract_text(self, image, bboxes):
        """
        Extracts the text from an image containing number plates.

        inputs:
            image: the image to extract text from
            bboxes: list of tuples representing bbox coordinates x1, y1, x2, y2
        """
        extracted_txts = []

        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            cropped_img = image[y1:y2, x1:x2]
            extracted_txt = self.recognise(cropped_img, x2 - x1)
            label = ''.join(extracted_txt).strip()
            extracted_txts.append(label)

        return extracted_txts

    def annotate_image(self, image):
        """
        Annotates the image with the predicted bbox and license number

        input: the image to extract text from
        returns: image with labels and annotated bboxes
        """
        # Details for annotating the image
        lw = max(round(sum(image.shape) / 2 * 0.003), 2)
        colour = (128, 128, 128)
        txt_colour = (255, 255, 255)
        tf = max(lw - 1, 1)
        sf = lw / 3

        # Extract the bboxes and labels from the image
        bboxes = self.extract_bboxes(image)
        labels = self.extract_text(image, bboxes)
        image_copy = np.copy(image)

        for bbox, label in zip(bboxes, labels):
            p1 = bbox[0], bbox[1]
            if label:
                w, h = cv2.getTextSize(label, 0, fontScale=sf, thickness=tf)[0]
                outside = p1[1] - h >= 3
                outside = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                annotated_image = cv2.rectangle(image_copy,
                                                p1,
                                                p2,
                                                colour,
                                                -1,
                                                cv2.LINE_AA)

                annotated_image = cv2.putText(annotated_image,
                                              label,
                                              (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                                              0,
                                              sf,
                                              txt_colour,
                                              thickness=tf,
                                              lineType=cv2.LINE_AA)

        return annotated_image

    def show_annotated_image(self, image):
        """
        Shows the annotated image.

        input: image - the image that will be annotated
        output: None
        """
        annotated_image = self.annotate_image(image)
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        plt.imshow(annotated_image)
