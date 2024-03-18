import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

PROJECT_NAME = "vizpool"
USERNAME = "Hassi34"

setuptools.setup(
    name=f"{PROJECT_NAME}",
    include_package_data = True,
    version="0.0.12.post3",
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
        "numpy >= 1.22",
        "pandas >= 1.2.0",
        "seaborn >= 0.12.2",
        "scikit-learn >= 1.3.1",
        "kaleido==0.2.1"
        ]
)   
