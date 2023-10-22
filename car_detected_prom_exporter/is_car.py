import argparse
import torch
import requests
import torchvision.transforms as transforms
import torchvision.models.detection as detection
from PIL import Image, ImageDraw
from io import BytesIO
from prometheus_client import start_http_server, Gauge, Enum, Summary, Counter
import cv2
import os
import numpy as np
import time


def cli():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--runonce", help="Run detection a single time and exit", action="store_true"
    )
    parser.add_argument("--show", help="Show images", action="store_true")
    parser.add_argument("--serve", help="Run Servers", action="store_true")
    args = parser.parse_args()

    if args.serve:
        polling_interval_seconds = int(os.getenv("POLLING_INTERVAL_SECONDS", "50"))
        exporter_port = int(os.getenv("EXPORTER_PORT", "9878"))

        app_metrics = AppMetrics(polling_interval_seconds=polling_interval_seconds)
        start_http_server(exporter_port)
        app_metrics.run_metrics_loop()

    run_detect(args)


def run_detect(args):
    while True:
        detect(model_setup(), show=args.show)
        if args.runonce:
            break


def model_setup():
    # Load the pre-trained Faster R-CNN model
    model = detection.fasterrcnn_resnet50_fpn(
        weights=detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    )
    # Run on GPU if cuda is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Moving model to {device}")
    model.to(device)
    model.eval()

    return model


def detect(model, show=False):
    result = {}

    cameras = {
        "192.168.2.154": {
            "masks": [
                [1347, 96, 1413, 174],
                [1354, 59, 1449, 139],
                [1458, 145, 1533, 183],
                [1425, 141, 1455, 165],
            ],
            "name": "crest_south",
        },
        "192.168.2.155": {
            "masks": [[35, 309, 64, 342], [5, 699, 287, 1063]],
            "name": "crest_north",
        },
    }

    URL = "http://{camera}/snap.jpeg"


    for camera, params in cameras.items():
        # Load an image and transform

        # Download the image
        response = requests.get(URL.format(camera=camera))
        response.raise_for_status()  # Raise an error for bad responses

        # Convert the response content to an image
        image = Image.open(BytesIO(response.content))

        draw = ImageDraw.Draw(image)

        # Mask neighbours
        for m in params["masks"]:
            draw.rectangle([m[0], m[1], m[2], m[3]], fill=(0, 0, 0))


        transform = transforms.Compose([transforms.ToTensor()])
        image_tensor = transform(image).unsqueeze(0)

        # Ensure the input tensor is in the same place as the model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        image_tensor = image_tensor.to(device)
        
        # Detect objects in the image
        with torch.no_grad():
            prediction = model(image_tensor)

        # Bring any detections back to Host Memory
         
        # Draw bounding boxes for detected cars

        pil_image_rgb = image.convert("RGB")
        opencv_image = np.array(pil_image_rgb)
        image_np = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)
        # image_np = cv2.imread(image)
        car_detected = 0
        for box, label, score in zip(
            prediction[0]["boxes"], prediction[0]["labels"], prediction[0]["scores"]
        ):
            if (
                label == 3 and score > 0.5
            ):  # 3 is the label for cars in COCO, and we consider detections with confidence > 0.5
                box = box.cpu().numpy().astype(int)
                print(box)
                cv2.rectangle(
                    image_np, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2
                )
                car_detected += 1

        if show:
            cv2.imshow("Detected Cars", image_np)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        result[params["name"]] = car_detected
        print("Camera: ", params["name"], " Car detected: ", car_detected)

    return result


class AppMetrics:
    """
    Representation of Prometheus metrics and loop to fetch and transform
    application metrics into Prometheus metrics.
    """

    def __init__(self, polling_interval_seconds=30):
        self.polling_interval_seconds = polling_interval_seconds
        self.last_message_type = 1
        # Prometheus metrics to collect
        self.crest_south = Gauge("crest_south", "Car Detected on Crest South")
        self.crest_north = Gauge("crest_north", "Car Detected on Crest North")
        self.detection_model = model_setup()

    def run_metrics_loop(self):
        """Metrics fetching loop"""
        while True:
            self.fetch()
            time.sleep(self.polling_interval_seconds)

    def fetch(self):
        """
        Get metrics from application and refresh Prometheus metrics with
        new values.
        """

        results = detect(self.detection_model)

        self.crest_south.set(results["crest_south"])
        self.crest_north.set(results["crest_north"])

        print("Fetching Result")
        print(results)


if __name__ == "__main__":
    cli()
