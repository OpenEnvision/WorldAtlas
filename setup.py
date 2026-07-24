from setuptools import find_namespace_packages, setup


setup(
    name="worldatlas-arena",
    version="1.0.0",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="stanford",
    packages=find_namespace_packages(include=["worldarena*"]),
    python_requires=">=3.8",
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
    package_data={"worldarena.benchmark.official_backends": ["LICENSE", "UPSTREAM.md"]},
    entry_points={
        "console_scripts": [
            "worldatlas-arena=worldarena.cli:main",
            "worldarena=worldarena.cli:main",
        ],
    },
)
