from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements  = f.read().splitlines()


setup(
    name="MLOps_Project_1",
    version="0.1",
    author="Vibhav Ahuja",
    packages=find_packages(),
    install_requires = requirements
)


