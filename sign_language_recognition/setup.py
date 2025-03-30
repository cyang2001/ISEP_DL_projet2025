"""
Setup script for the Sign Language Recognition package.
"""
from setuptools import setup, find_packages

setup(
    name="sign_language_recognition",
    version="0.1.0",
    description="Deep learning-based video sign language recognition system",
    author="ISEP DL Project Team",
    author_email="chen.yang@isep.eleve.fr",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.10, <3.13",
    install_requires=[
        # Core dependencies will come from requirements.txt
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
) 