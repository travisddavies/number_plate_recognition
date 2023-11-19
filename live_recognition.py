from engine.recognition import NumberPlateRecogniser
import cv2


def main():
    # Number plate recognition model
    model = NumberPlateRecogniser(size='n')
    # Access the camera
    camera = cv2.VideoCapture(0)
    # Start a loop that won't break until the window is quit
    while (cv2.waitKey(1) == -1):
        # Read the frames of the camera
        success, frame = camera.read()
        if success:
            # Get the bboxes of the number plates
            bboxes = model.extract_bboxes(frame)
            # Get the number plate numbers
            labels = model.extract_text(frame, bboxes)
            # Create an annotated version of the frame with the bbox and label
            annotated_frame = model.annotate_all_in_one(frame, bboxes, labels)
            # Show the annotated frame to the screen
            cv2.imshow('Number Plate Recognition', annotated_frame)

        if 0xff == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
