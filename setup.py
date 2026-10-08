from pathlib import Path

from setuptools import find_namespace_packages, setup


setup(
    name="worldatlas",
    version="1.0.0",
    description="WorldAtlas benchmark and model adapters for world generation",
    long_description=Path(__file__).with_name("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="OpenEnvision",
    url="https://github.com/OpenEnvision/WorldAtlas",
    license="MIT",
    packages=["worldatlas"] + [
        f"worldatlas.{name}"
        for name in find_namespace_packages(
            where="worldatlas",
            exclude=[
                "docs", "docs.*", "datasets", "datasets.*",
                "tests", "tests.*", "*.tests", "*.tests.*", "*.__pycache__", "*.backup*",
            ],
        )
    ],
    python_requires=">=3.10",
    install_requires=[
        "av",
        "python-dotenv",
        "omegaconf",
        "submitit",
        "structlog",
        "pyfiglet",
        "numpy",
        "pandas",
        "pillow",
        "plotly",
        "PyYAML",
        "opencv-python",
        "scipy",
        "tabulate",
        "fire",
        "hydra-core",
        "requests",
    ],
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "worldatlas=worldatlas.cli:main",
        ],
    },
)
