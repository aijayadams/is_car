from setuptools import setup, find_packages

setup(
    name='car_detected_prom_exporter',
    version='0.1',
    packages=find_packages(),
    install_requires=[
       "torch", 
       "torchvision",
       "opencv-python",
       "prometheus_client"

    ],
)
