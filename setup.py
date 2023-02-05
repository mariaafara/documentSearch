import os
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


with open("requirements.txt") as f:
    install_requires = f.read().strip().split("\n")

setup(
    name="documentSearch",
    version="0.0.1",
    author="Maria Afara",
    author_email="maria-afara5@hotmail.com",
    description="Document search engine.",
    packages=find_packages(),
    include_package_data=True,
    long_description=read('README.md'),
    install_requires=install_requires,
)
