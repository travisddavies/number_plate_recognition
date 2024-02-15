from engine.recognition import NumberPlateRecogniser
import cv2
from argparse import ArgumentParser
import os


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
    parser.add_argument('-d', '--directory_mode', type=bool,
                              help='Whether the input and output are directories')
    parser.add_argument('-p', '--pi', type=bool,
                        help='Whether or not the device running the program is a raspberry pi')
    args = parser.parse_args()

    images = get_images(args.input, args.directory_mode)
    process_image(images, args.output, args.directory_mode, args.pi)


def get_images(input, directory_mode):
    assert input, 'Must provide an input image path.'
    if directory_mode:
        files = os.listdir(input)
        filepaths = [os.path.join(input, file) for file in files]
        return filepaths

    return [input]


def process_image(input_filepaths, output, directory_mode, pi):
    assert output, 'Must provide an output image path.'
    # Number plate recognition model
    model = NumberPlateRecogniser()

    for i, file in enumerate(input_filepaths):
        image = cv2.imread(file)
        # Get the bboxes of the number plates
        bboxes = model.extract_bboxes(image)
        # Get the number plate numbers
        labels = model.extract_text(image, bboxes, pi)
        # Create an annotated version of the frame with the bbox and label
        annotated_image = model.annotate_all_in_one(image, bboxes, labels)
        # Save the frame to a video
        if directory_mode:
            output_name = f'{i}.jpg'
            output_path = os.path.join(output, output_name)
            cv2.imwrite(output_path, annotated_image)
        else:
            cv2.imwrite(output, annotated_image)


if __name__ == "__main__":
    main()
