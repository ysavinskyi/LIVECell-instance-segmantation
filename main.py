import argparse
from client.server_requests import send_image
from client.image_processing import process_prediction, resize_image
from client.plotting import plot_images


def main(image_path):
    response_json = send_image(image_path)
    prediction_list = response_json['predictions']
    prediction_image = process_prediction(prediction_list)
    original_image = resize_image(image_path)
    plot_images(original_image, prediction_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Send an image to the server for segmentation and display results.')
    parser.add_argument('image_path', type=str, help='Path to the image file to be sent to the server.')
    args = parser.parse_args()
    main(args.image_path)
