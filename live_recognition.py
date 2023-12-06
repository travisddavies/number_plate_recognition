from engine.recognition import NumberPlateRecogniser
import cv2
import asyncio
from data_collection import NumberPlateCollector


async def main():
    """
    Iterates through the frames of a local video or live camera footage.

    If a live camera feed is iterated through, an annotated video will be shown
    on the screen, but no video will be saved.
    """
    # If it's live mode, go through a live mode of number plate recognition
    await perform_live_mode()

    # Kill the open window
    cv2.waitKey(1000)
    cv2.destroyAllWindows()


async def perform_live_mode():
    # Number plate recognition model
    model = NumberPlateRecogniser(size='n')
    # Data collector for number plates
    collector = NumberPlateCollector()
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
            collector.collect_data(frame, labels, bboxes)
        else:
            break
        # Kill the open window
        k = cv2.waitKey(1) & 0xFF
        if k in {27, ord('q')}:
            break

    # Stop the camera
    camera.release()
    task = asyncio.create_task(collector.send_to_db())
    await task


if __name__ == "__main__":
    asyncio.run(main())
