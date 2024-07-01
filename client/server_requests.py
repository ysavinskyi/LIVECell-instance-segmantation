import requests


def send_image(image_path):
    url = 'http://127.0.0.1:8000/segmentation/predict/'

    with open(image_path, 'rb') as image_file:
        files = {'file': image_file}
        response = requests.post(url, files=files)

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Error: {response.json()}")