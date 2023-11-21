from engine.recognition import NumberPlateRecogniser
import cv2
from argparse import ArgumentParser


def main():
    parser = ArgumentParser(description='Process an image file.')
    parser.add_argument('-i', '--input', type=str, help='Path to input video.')
    parser.add_argument('-o', '--output', type=str,
                              help='Path to output annotated video.')
    parser.add_argument('-l', '--live_mode', type=bool,
                              help='Non-optional - True to access camera and annotate live footage, false to annotate local video')

    args = parser.parse_args()

    if args.live_mode:
        perform_live_mode()
    else:
        process_video(args.input, args.output)

    cv2.waitKey(1000)
    cv2.destroyAllWindows()


def perform_live_mode():
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

        k = cv2.waitKey(1) & 0xFF
        if k in {27, ord('q')}:
            break

    camera.release()
    cv2.waitKey(1000)
    cv2.destroyAllWindows()


def process_video(input_filepath, output_filepath):
    assert input_filepath, 'Must provide an input video path.'
    assert output_filepath, 'Must provide an output video path.'
    model = NumberPlateRecogniser(size='n')
    camera = cv2.VideoCapture(input_filepath)
    width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = camera.get(cv2.CAP_PROP_FPS)
    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_filepath, fourcc, fps, (width, height))

    while (cv2.waitKey(1) == -1):
        success, frame = camera.read()
        if success:
            bboxes = model.extract_bboxes(frame)
            labels = model.extract_text(frame, bboxes)
            annotated_frame = model.annotate_all_in_one(frame, bboxes, labels)
            cv2.imshow('Number Plate Recognition', annotated_frame)
            out.write(annotated_frame)
        else:
            break

        k = cv2.waitKey(1) & 0xFF
        if k in {27, ord('q')}:
            break

    out.release()
    camera.release()


if __name__ == "__main__":
    main()
