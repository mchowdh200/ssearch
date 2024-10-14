from setuptools import find_packages, setup

setup(
    name="ssearch",
    version="0.1",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "ssearch = ssearch.__main__:main",
        ],
    },
)
