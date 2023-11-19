from engine.recognition import NumberPlateRecogniser
import cv2


def main():
    model = NumberPlateRecogniser(size='n')
    camera = cv2.VideoCapture(0)
    while (cv2.waitKey(1) == -1):
        success, frame = camera.read()
        if success:
            bboxes = model.extract_bboxes(frame)
            labels = model.extract_text(frame, bboxes)
            annotated_frame = model.annotate_all_in_one(frame, bboxes, labels)
            cv2.imshow('Number Plate Recognition', annotated_frame)

        if 0xff == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
