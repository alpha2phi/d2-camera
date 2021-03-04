import sys
from setuptools import setup, find_packages

sys.path[0:0] = ["d2_camera"]
from version import __version__

setup(
    name="d2-camera",
    packages=find_packages(),
    include_package_data=True,
    entry_points={"console_scripts": ["d2 = d2_camera.cli:main",],},
    version=__version__,
    license="MIT",
    description="D2 Camera",
    author="alpha2phi",
    author_email="alpha2phi@gmail.com",
    url="https://github.com/alpha2phi/d2-camera",
    keywords=[
        "artificial intelligence",
        "deep learning",
        "object detection",
        "segmentation",
    ],
    install_requires=[
        "einops",
        "fire",
        "ftfy",
        "torch",
        "torch_optimizer",
        "torchvision",
        "tqdm",
        "regex",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
    ],
)
