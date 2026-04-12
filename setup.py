from setuptools import setup, find_packages

setup(
    name="rams",
    version="0.1.0",
    description="Resource-Adaptive Model Switching for Edge AI",
    authors=[
        {"name": "Kushal Khemani"},
        {"name": "Ajinkyaa Lokhande"},
    ],
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "psutil>=5.9",
        "pyyaml>=6.0",
    ],
    extras_require={
        "inference": ["ultralytics>=8.0", "torch>=2.0", "torchvision>=0.15", "numpy>=1.24"],
    },
)
