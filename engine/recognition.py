from ultralytics import YOLO
import cv2
import numpy as np
from paddleocr import PaddleOCR
import logging
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger('paddle')
logger.setLevel(logging.WARN)


class NumberPlateRecogniser:
    """
    A number plate recognition model that both detects number plates
    and extracts the text from the detected number plates.

    For initialisation, choose either size='n' or size='s'.
    """

    def __init__(self, country):
        # Number plate detection model.
        self._detector = YOLO('./engine/best_weights/yolov8n/best.pt')
        # OCR model for number plate text extraction.
        # If Australia was chosen, upload the australian weights,
        # else upload chinese weights.
        if country == 'au':
            rec_model_dir = './engine/best_weights/aus_number_plates'
            self._recogniser = PaddleOCR(use_angle_cls=True,
                                         detect=True, rec=True,
                                         rec_model_dir=rec_model_dir,
                                         show_log=False)
        else:
            rec_model_dir = './engine/best_weights/chinese_number_plates/Teacher'
            rec_char_dict_path = './engine/chinese_number_plate.txt'
            self._recogniser = PaddleOCR(use_angle_cls=True,
                                         detect=True, rec=True,
                                         rec_model_dir=rec_model_dir,
                                         rec_char_dict_path=rec_char_dict_path,
                                         show_log=False)

    # Detects the number plate in an image.
    def detect(self, image):
        return self._detector(image)

    def recognise(self, cropped_image):
        """
        Recognises the number plate text of a cropped_image.

        Input: cropped_image - cropped to number plate

        Returns: the text of the number plate as a list
        """
        return self._recogniser.ocr(cropped_image)

    def extract_bboxes(self, image):
        """
        Extracts the bboxes of the given image that are predicted to be number
        plates.

        Input: the image to extract text from

        Returns: list of tuples representing bboxe coordinates x1, y1, x2, y2
        """
        predictions = self.detect(image)
        extracted_bboxes = []

        for prediction in predictions:
            boxes = prediction.boxes
            for box in boxes:
                xyxy = box.xyxy.squeeze()
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), \
                    int(xyxy[3])
                extracted_bboxes.append([x1, y1, x2, y2])

        return extracted_bboxes

    def extract_text(self, image, bboxes):
        """
        Extracts the text from an image containing number plates.

        Inputs:
            image: the image to extract text from
            bboxes: list of tuples representing bbox coordinates x1, y1, x2, y2

        Returns: The predicted license plate number as a string
        """
        labels = []
        for bbox in bboxes:
            good_results = []
            label = ''
            x1, y1, x2, y2 = bbox
            cropped_img = image[y1:y2, x1:x2]
            predictions = self.recognise(cropped_img)
            for prediction in predictions:
                if not prediction:
                    continue

                for line in prediction:
                    y1 = line[0][0][1]
                    y2 = line[0][1][1]
                    y3 = line[0][2][1]
                    y4 = line[0][3][1]
                    y_coords = [y1, y2, y3, y4]
                    if self._spreads_over_numplate(cropped_img, y_coords):
                        good_results.append(line)
                    ordered_results = sorted(good_results, key=lambda x: x[0][0])

                if self._is_confident_prediction(ordered_results):
                    for label_subtext in ordered_results:
                        label += label_subtext[1][0]
                labels.append(label)

        return labels

    def _is_confident_prediction(self, ordered_results):
        """
        Determines whether the confidence of the predicted number plate is over
        0.9

        Input: The license plate numbers as arrays, ordered by their placement

        Returns: True if over 0.9 confidence, false otherwise
        """
        is_confident_count = 0
        for result in ordered_results:
            if result[1][1] < 0.75:
                is_confident_count += 1
        return is_confident_count == 0

    def _spreads_over_numplate(self, num_plate, y_coords):
        y_min = min(y_coords)
        y_max = max(y_coords)
        y1, y2, y3, y4 = y_coords
        h = y4 - y1
        return (y_min < num_plate.shape[0] / 2 and y_max > num_plate.shape[0] / 2) \
            and h > 0.3 * num_plate.shape[0]

    def annotate(self, image, bboxes, labels):
        """
        Annotates the image with the predicted bbox and license number

        Input: the image to extract text from

        Returns: image with labels and annotated bboxes
        """
        # Details for annotating the image
        lw = max(round(sum(image.shape) / 2 * 0.003), 2)
        colour = (128, 128, 128)
        txt_colour = (255, 255, 255)
        tf = max(lw - 1, 1)
        sf = lw / 3

        # Extract the bboxes and labels from the image
        image_copy = np.copy(image)
        annotated_images = []

        # Create a list of individually annotated images, one for each car in
        # image.
        for bbox, label in zip(bboxes, labels):
            p1 = bbox[0], bbox[1]
            if label:
                w, h = cv2.getTextSize(label, 0, fontScale=sf, thickness=tf)[0]
                outside = p1[1] - h >= 3
                outside = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                p3 = bbox[2], bbox[3]
                # The bbox for the detected license plate
                annotated_image = cv2.rectangle(image_copy,
                                                p1,
                                                p3,
                                                colour,
                                                20)

                # The filled in box where the license plate number will be
                annotated_image = cv2.rectangle(annotated_image,
                                                p1,
                                                p2,
                                                colour,
                                                -1,
                                                cv2.LINE_AA)

                # The license plate number annotated on the image
                annotated_image = cv2.putText(annotated_image,
                                              label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, sf, txt_colour,
                                              thickness=tf,
                                              lineType=cv2.LINE_AA)

                annotated_images.append(annotated_image)

        return annotated_images

    def annotate_all_in_one(self, image, bboxes, labels):
        """
        Gives an image with all the annotations of multiple license plates
        in the given image

        Input:
            - The image that will be annotated
            - The bboxes of the predicted license plate locations
            - The numbers of each license plate

        Returns: The annotated image with bboxes and license plate numbers
        """
        lw = max(round(sum(image.shape) / 2 * 0.003), 2)
        colour = (128, 128, 128)
        txt_colour = (255, 255, 255)
        tf = max(lw - 1, 1)
        sf = lw / 3

        # Extract the bboxes and labels from the image
        annotated_image = np.copy(image)

        for bbox, label in zip(bboxes, labels):
            p1 = bbox[0], bbox[1]
            if label:
                w, h = cv2.getTextSize(label, 0, fontScale=sf, thickness=tf)[0]
                outside = p1[1] - h >= 3
                outside = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                p3 = bbox[2], bbox[3]

                # The bbox for the detected license plate
                annotated_image = cv2.rectangle(annotated_image,
                                                p1,
                                                p3,
                                                colour,
                                                2)

                # The filled in box where the license plate number will be
                annotated_image = cv2.rectangle(annotated_image,
                                                p1,
                                                p2,
                                                colour,
                                                -1,
                                                cv2.LINE_AA)

                # The license plate number annotated on the image
                text_position = (p1[0], p1[1] - 30 if outside else p1[1] + h + 30)
                annotated_image = self._add_text(
                    label, annotated_image, text_position,
                    tf, txt_colour, outside)

        return annotated_image

    def _add_text(self, label, image, text_position, tf, txt_colour, outside):
        # Assuming you have the variables label, p1, outside, h, sf,
        # txt_colour, tf defined
        # Convert the annotated_image from OpenCV format (BGR) to Pillow format
        # (RGB)
        annotated_image = Image.fromarray(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Create a Pillow ImageDraw object
        draw = ImageDraw.Draw(annotated_image)

        # Convert the OpenCV text color format (BGR) to Pillow format (RGB)
        text_color_pil = tuple(reversed(txt_colour))

        # Load a font (you may need to adjust the font path)
        font_path = "/usr/share/fonts/wenquanyi/wqy-microhei/wqy-microhei.ttc"
        font_size = 30
        font = ImageFont.truetype(font_path, font_size)

        # Annotate the image with text
        draw.text(text_position, label, font=font, fill=text_color_pil)

        # Convert the Pillow image back to OpenCV format (BGR)
        annotated_image_cv2 = cv2.cvtColor(
            np.array(annotated_image), cv2.COLOR_RGB2BGR)

        return annotated_image_cv2
