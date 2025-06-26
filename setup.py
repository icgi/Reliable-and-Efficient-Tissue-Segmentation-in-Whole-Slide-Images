from setuptools import setup, find_packages

setup(
    name='nnunetv2',
    version='2.5.1',  # Replace with your actual version
    packages=find_packages(exclude=['dockerfiles', 'dockerfiles.*']),
    install_requires=[
        # List your package dependencies here
        'batchgenerators>=0.25',
        'torch>=1.10.0',
        'numpy',
        # ... other dependencies
    ],
    python_requires='>=3.6'
)
