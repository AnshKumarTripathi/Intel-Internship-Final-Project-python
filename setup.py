from setuptools import setup, find_packages

setup(
    name="helmet-detection-system",
    version="1.0.0",
    description="AI-powered helmet detection system",
    author="Your Name",
    packages=find_packages(),
    python_requires=">=3.8,<3.13",
    install_requires=[
        "streamlit==1.28.1",
        "tensorflow==2.15.0",
        "opencv-python==4.8.1.78",
        "pillow==10.0.1",
        "numpy==1.24.3",
        "matplotlib==3.7.2",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
) 