"""
    Probably broken.
"""
from setuptools import setup

setup(
    py_modules = ['microarray.py'],
    # package_data = {'':['*.csv']},
    entry_points={
        'console_scripts': [
            # 'microarray-learn = microarray.main'
        ]},

    # Some packages were needed to compile this libraries on a debian machine:
    #
    # sudo aptitude install build-essential gfortran liblapack-dev libblas-dev \
    #                       libfreetype-dev
    #
    # Maybe more...
    install_requires=[
        "scikit-learn",
        "matplotlib",
        "numpy",
        "scipy",
        "pandas",
        "docopt"],

    name='microarray',
    version='0.1',
    author="Pedro Sousa Lacerda",
)
