from setuptools import setup
from setuptools import find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="cypher",
    version="2.0.0",
    author="Tanish Shinde",
    author_email="simon.blanke@yahoo.com",
    license="MIT",
    description="A hyperparameter optimization toolbox for convenient and fast prototyping",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords=["machine learning", "deep learning", "optimization", "data-science"],
    url="https://github.com/TanishShinde/cypher",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
    ],
    install_requires=["numpy", "pandas", "tqdm"],
)
