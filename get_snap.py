import requests
from PIL import Image
from io import BytesIO
import time
import os


cameras = ["192.168.2.154", "192.168.2.155"]

# Ensure there's a directory to save the images
OUTPUT_DIR = 'downloaded_images'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

URL = "http://{camera}/snap.jpeg"

while True:
    for camera in cameras:
        # Download the image
        response = requests.get(URL.format(camera=camera))
        response.raise_for_status()  # Raise an error for bad responses

        # Convert the response content to an image
        image = Image.open(BytesIO(response.content))

        # Get the current timestamp and format it
        timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
        file_path = os.path.join(OUTPUT_DIR, f"{camera}.{timestamp}.jpg")

        # Save the image
        image.save(file_path)
        print(f"Downloaded image to {file_path}")

    # Wait for 1 minute
    time.sleep(60)