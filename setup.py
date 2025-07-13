from setuptools import setup, find_packages

setup(
    name='leukemia_detection',
    version='0.1.0',
    author='Sanchit Chauhan',
    description='A deep learning pipeline for leukemia cell image classification using CNN and transfer learning.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/leukemia-detection',  # Replace with your repo URL if available
    packages=find_packages(exclude=["tests", "scripts", "notebooks"]),
    include_package_data=True,
    install_requires=[
        'tensorflow>=2.10.0',
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'PyYAML',
        'opencv-python',
    ],
    extras_require={
        'dev': [
            'pytest',
            'flake8',
            'jupyter',
        ]
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
    ],
    python_requires='>=3.7',
)