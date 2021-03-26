from setuptools import setup

exec(open('glomtf/version.py').read())

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="glom-tf",
    version="0.0.1",
    description="Implement GLOM, part-whole hierarchies in TensorFlow",
    packages=["glomtf"],

    long_description=long_description,
    long_description_content_type="text/markdown",

    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Mathematics"
    ],

    url="https://github.com/Rishit-dagli/GLOM-TensorFlow",
    author="Rishit Dagli",
    author_email="rishit.dagli@gmail.com",

    install_requires=[
        "tensorflow >= 2.2.0",
        "einops>=0.3",
    ],

    extras_require={
        "dev": [
            "check-manifest",
            "twine",
            "numpy"
        ],
    },
)
