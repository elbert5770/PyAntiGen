from setuptools import setup, find_packages

setup(
    name='pyantigen',
    # No version here. It is declared dynamic in pyproject.toml and supplied by
    # setuptools-scm from the git tag; setting it in both places is a build
    # error. This file now only carries package_data, which pyproject does not.
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
