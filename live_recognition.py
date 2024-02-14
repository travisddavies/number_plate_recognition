from engine.recognition import NumberPlateRecogniser
import cv2
import asyncio
from data_collection import NumberPlateCollector
from picamera2 import Picamera2
from argparse import ArgumentParser


async def main():
    """
    Iterates through the frames of a local video or live camera footage.

    If a live camera feed is iterated through, an annotated video will be shown
    on the screen, but no video will be saved.
    """
    # Arguments for the program
    parser = ArgumentParser(description='Processing live feed')
    # Input filepath for the model to read from and annotate
    parser.add_argument('-p', '--pi', type=bool,
                        help='Whether or not the device running the program is a raspberry pi')
    args = parser.parse_args()
    # If it's live mode, go through a live mode of number plate recognition
    await perform_live_mode(args.pi)

    # Kill the open window
#    cv2.waitKey(1000)
#    cv2.destroyAllWindows()


async def perform_live_mode(pi):
    # Number plate recognition model
    model = NumberPlateRecogniser()
    # Data collector for number plates
    collector = NumberPlateCollector()
    # Access the camera
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(
        main={"format": 'RGB888'}))
    picam2.start()
    # Start a loop that won't break until the window is quit
    while (cv2.waitKey(1) == -1):
        # Read the frames of the camera
        frame = picam2.capture_array()
        # Get the bboxes of the number plates
        bboxes = model.extract_bboxes(frame)
        # Get the number plate numbers
        labels = model.extract_text(frame, bboxes, pi)
        # Create an annotated version of the frame with the bbox and label
#        annotated_frame = model.annotate_all_in_one(frame, bboxes, labels)
        # Show the annotated frame to the screen
#        cv2.imshow('Number Plate Recognition', annotated_frame)
        collector.collect_data(frame, labels, bboxes)
        # Kill the open window
        # k = cv2.waitKey(1) & 0xFF
        # if k in {27, ord('q')}:
            # break

#    task = asyncio.create_task(collector.send_to_db())
#    await task

if __name__ == "__main__":
    asyncio.run(main())
