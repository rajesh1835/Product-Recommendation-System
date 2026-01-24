from setuptools import setup, find_packages

setup(
    name="recsys-amz",
    version="0.1.0",
    author="Tarigonda Rajesh",
    author_email="rajeshtarigonda@example.com",
    description="E-Commerce Recommendation System using Ratings and Reviews",
    packages=find_packages(exclude=["venv", "venv.*", "notebooks"]),
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "matplotlib",
        "seaborn",
        "nltk",
        "scikit-surprise",
    ],
)