import os

from setuptools import find_packages, setup

this_dir = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_dir, "requirements.txt"), "r") as f:
    requirements = [line.strip() for line in f.readlines()]

setup(
    name="ddcfr",
    version="1.0.0",
    description="Solving Imperfect-Information Games via Dynamic Discounted Regret Minimization",
    packages=find_packages(exclude=("tests*", "docs*", "examples*")),
    zip_safe=True,
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)
