from engine.recognition import NumberPlateRecogniser
import cv2
from argparse import ArgumentParser


def main():
    """
    Iterates through the frames of a local video or live camera footage.

    If a video is iterated through, an annotated version of the video will be
    saved at a designated filepath.

    If a live camera feed is iterated through, an annotated video will be shown
    on the screen, but no video will be saved.
    """
    # Arguments for the program
    parser = ArgumentParser(description='Process an image file.')
    # Input filepath for the model to read from and annotate
    parser.add_argument('-i', '--input', type=str, help='Path to input video.')
    # Output filepath for the model to write to and annotate
    parser.add_argument('-o', '--output', type=str,
                              help='Path to output annotated video.')
    # Flag for whether to read from local video or open computer camera
    parser.add_argument('-l', '--live_mode', type=bool,
                              help='Non-optional - True to access camera and annotate live footage, false to annotate local video')

    args = parser.parse_args()

    # If it's live mode, go through a live mode of number plate recognition
    if args.live_mode:
        perform_live_mode()
    # Otherwise, go through a local video and save the output
    else:
        process_video(args.input, args.output)

    # Kill the open window
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

        # Kill the open window
        k = cv2.waitKey(1) & 0xFF
        if k in {27, ord('q')}:
            break

    # Stop the camera
    camera.release()


def process_video(input_filepath, output_filepath):
    assert input_filepath, 'Must provide an input video path.'
    assert output_filepath, 'Must provide an output video path.'
    # Number plate recognition model
    model = NumberPlateRecogniser(size='n')
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
            labels = model.extract_text(frame, bboxes)
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
