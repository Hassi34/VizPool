import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

PROJECT_NAME = "vizpool"
USERNAME = "Hassi34"

setuptools.setup(
    name=f"{PROJECT_NAME}",
    version="0.0.8.dev1",
    author= USERNAME,
    author_email="hasnainmehmood3435@gmail.com",
    description="A high-level python API with low-code support for Data Visualization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{USERNAME}/{PROJECT_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{USERNAME}/{PROJECT_NAME}/issues",
    },
    classifiers=[
            'Intended Audience :: Science/Research',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'License :: OSI Approved :: MIT License',
            'Topic :: Scientific/Engineering :: Visualization',
            'Operating System :: OS Independent'
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.7",
    install_requires = [
        "plotly >= 5.8.0",
        "numpy >= 1.22.4",
        "pandas >= 1.3.5",
        "seaborn >= 0.11.1",
        "scikit-learn==1.1.2",
    ]
)   