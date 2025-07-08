from setuptools import setup, find_packages
import io

setup(
    name="Convention",
    version="0.1.0",
    author="LiuBai",
    description="A python version code repository for implementing the agreements and implementations in the Convention-Template.",
    long_description=io.open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/NINEMINEsigma/Convention-Python",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.12",
    install_requires=[
        "colorama",
        "pydantic"
    ],
    exclude_package_data={"": ["*.meta"]},
)