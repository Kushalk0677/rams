from setuptools import find_packages, setup

setup(
    name="rams",
    version="0.1.0",
    description="Resource-Adaptive Model Switching for Edge AI",
    author="Kushal Khemani, Evan Leri, George Xu",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "psutil>=5.9",
        "PyYAML>=6.0",
        "numpy>=1.24",
    ],
    extras_require={
        "plots": [
            "matplotlib>=3.8",
        ],
        "inference": [
            "opencv-python>=4.8",
            "onnxruntime>=1.17",
            "ultralytics>=8.0",
            "torch>=2.0",
            "torchvision>=0.15",
        ],
        "full": [
            "matplotlib>=3.8",
            "opencv-python>=4.8",
            "onnxruntime>=1.17",
            "ultralytics>=8.0",
            "torch>=2.0",
            "torchvision>=0.15",
        ],
    },
)
