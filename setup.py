from setuptools import setup, find_packages

setup(
    name='pyantigen',
    version='0.1.0',
    description='A declarative framework for building compartmental Antimony models',
    author='Open Source Contributor',
    packages=find_packages(include=['framework*']),
    install_requires=[
        'tellurium',
        'matplotlib',
        'numpy'
    ],
    entry_points={
        'console_scripts': [
            'pyantigen-create=framework.cli:create_project',
        ],
    },
)
