from pathlib import Path

import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


__version__ = "0.1.0"

REPO_NAME = "Machine-Learning"
AUTHOR_USER_NAME = "JannisEbling"
SRC_REPO = "MLOpsLib"
AUTHOR_EMAIL = "jannis.ebling@outlook.de"
REQUIREMENTS_PATH = Path(__file__).parent / "requirements.txt"


def parse_requirements(filename):
    with open(filename, "r") as f:
        return f.read().splitlines()


setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A small python package for ml app",
    long_description=long_description,
    long_description_content="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "MLOpsLib"},
    entry_points={
        "console_scripts": [
            "mlopslib=bin.cli:main",
        ],
    },
    packages=setuptools.find_packages(where="MLOpsLib"),
    install_requires=parse_requirements(REQUIREMENTS_PATH),
)
