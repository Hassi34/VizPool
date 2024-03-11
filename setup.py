import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

PROJECT_NAME = "vizpool"
USERNAME = "Hassi34"

setuptools.setup(
    name=f"{PROJECT_NAME}",
    include_package_data = True,
    version="0.0.11",
    license='MIT',
    author= "Hasanain Mehmood",
    author_email="hasanain@aicaliber.com",
    description="A Python Library with Low-Code support for Basic to Advance Static & Interactive Visualizations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{USERNAME}/{PROJECT_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{USERNAME}/{PROJECT_NAME}/issues",
    },
    classifiers=[
            'Development Status :: 5 - Production/Stable',
            'Intended Audience :: Science/Research',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10',
            'Programming Language :: Python :: 3.11',
            'Programming Language :: Python :: 3.12',
            'License :: OSI Approved :: MIT License',
            'Topic :: Scientific/Engineering :: Visualization',
            'Operating System :: OS Independent'
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.7",
    install_requires = [
        "plotly >= 5.19.0",
        "numpy >= 1.26.4",
        "pandas >= 2.2.1",
        "seaborn >= 0.13.2",
        "scikit-learn==1.4.0",
        "kaleido==0.2.1"
    ]
)   