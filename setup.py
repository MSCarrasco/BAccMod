from setuptools import setup, find_packages
from os import path

# read the contents of your README file
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='BAccMod',
    packages=find_packages(),
    version='0.2.0',
    license='LGPL v3',
    description='Calculate 2D and 3d acceptance model for creating maps with gammapy (IACT analysis)',
    url='https://github.com/mdebony/BAccMod',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=[
        'numpy',
        'scipy',
        'astropy>=4.0',
        'regions>=0.7,<0.11'
    ],
    author='Mathieu de Bony de Lavergne, Gabriel Emery, Marie-Sophie Carrasco',
    author_email='debony@cppm.in2p3.fr'
)
