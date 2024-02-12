from engine.recognition import NumberPlateRecogniser
import cv2
import asyncio
from data_collection import NumberPlateCollector
from argparse import ArgumentParser
from picamera2 import Picamera2


async def main():
    """
    Iterates through the frames of a local video or live camera footage.

    If a live camera feed is iterated through, an annotated video will be shown
    on the screen, but no video will be saved.
    """
    # Arguments for the program
    parser = ArgumentParser(description='Process an image file.')
    parser.add_argument('-c', '--country', type=str,
                              help='Country of number plates - either au or ch')

    args = parser.parse_args()
    # If it's live mode, go through a live mode of number plate recognition
    await perform_live_mode(args.country)

    # Kill the open window
    cv2.waitKey(1000)
    cv2.destroyAllWindows()


async def perform_live_mode(country):
    assert country in ['au', 'ch'], 'country must either be au or ch'
    # Number plate recognition model
    model = NumberPlateRecogniser(country)
    # Data collector for number plates
    collector = NumberPlateCollector()
    # Access the camera
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(
        main={"format": 'XRGB8888', "size": (640, 480)}))
    picam2.start()
    # Start a loop that won't break until the window is quit
    while (cv2.waitKey(1) == -1):
        # Read the frames of the camera
        frame = picam2.capture_array()
        success = True
        if success:
            # Get the bboxes of the number plates
            bboxes = model.extract_bboxes(frame)
            # Get the number plate numbers
            labels = model.extract_text(frame, bboxes)
            # Create an annotated version of the frame with the bbox and label
            annotated_frame = model.annotate_all_in_one(frame, bboxes, labels)
            # Show the annotated frame to the screen
            cv2.imshow('Number Plate Recognition', annotated_frame)
            collector.collect_data(frame, labels, bboxes)
        else:
            break
        # Kill the open window
        k = cv2.waitKey(1) & 0xFF
        if k in {27, ord('q')}:
            break

#    task = asyncio.create_task(collector.send_to_db())
#    await task

if __name__ == "__main__":
    asyncio.run(main())
