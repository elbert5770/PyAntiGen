from setuptools import setup, find_packages

setup(
    name='pyantigen',
    version='1.0.7',
    description='A declarative framework for building compartmental Antimony models',
    author='Open Source Contributor',
    packages=find_packages(include=['framework*']),
    package_data={
        'framework': [
            'template/Example/*.py',
            'template/Example/Modules/*.py',
            'template/data/*.csv',
        ],
    },
    install_requires=[
        'tellurium',
        'matplotlib',
        'numpy',
        'pandas',
        'scipy',
        'pypesto'
    ],
    entry_points={
        'console_scripts': [
            'pyantigen-create=framework.cli:create_project',
        ],
    },
)
