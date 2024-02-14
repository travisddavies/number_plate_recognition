from engine.recognition import NumberPlateRecogniser
import cv2
from argparse import ArgumentParser


def main():
    """
    Iterates through the frames of a local video or live camera footage.

    If a video is iterated through, an annotated version of the video will be
    saved at a designated filepath.
    """
    # Arguments for the program
    parser = ArgumentParser(description='Process an image file.')
    # Input filepath for the model to read from and annotate
    parser.add_argument('-i', '--input', type=str, help='Path to input video.')
    # Output filepath for the model to write to and annotate
    parser.add_argument('-o', '--output', type=str,
                              help='Path to output annotated video.')
    parser.add_argument('-p', '--pi', type=bool,
                        help='Whether or not the device running the program is a raspberry pi')
    args = parser.parse_args()

    process_video(args.input, args.output, args.pi)

    # Kill the open window
    cv2.waitKey(1000)
    cv2.destroyAllWindows()


def process_video(input_filepath, output_filepath, pi):
    assert input_filepath, 'Must provide an input video path.'
    assert output_filepath, 'Must provide an output video path.'
    # Number plate recognition model
    model = NumberPlateRecogniser()
    # Access the camera
    camera = cv2.VideoCapture(input_filepath)
    width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = camera.get(cv2.CAP_PROP_FPS)
    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_filepath, fourcc, fps, (width, height))

    # Start a loop that won't break until the window is quit
    while (cv2.waitKey(1) == -1):

        success, frame = camera.read()
        if success:
            # Get the bboxes of the number plates
            bboxes = model.extract_bboxes(frame)
            # Get the number plate numbers
            labels = model.extract_text(frame, bboxes, pi)
            # Create an annotated version of the frame with the bbox and label
            annotated_frame = model.annotate_all_in_one(frame, bboxes, labels)
            # Show the annotated frame to the screen
            cv2.imshow('Number Plate Recognition', annotated_frame)
            # Save the frame to a video
            out.write(annotated_frame)
        else:
            break

        # If the 'q' button or 'ESC' button are pressed, break the while loop
        k = cv2.waitKey(1) & 0xFF
        if k in {27, ord('q')}:
            break

    # Stop the camera and saving of the video
    out.release()
    camera.release()


if __name__ == "__main__":
    main()
