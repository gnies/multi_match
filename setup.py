from setuptools import setup
import os

ROOT = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(ROOT, 'README.md'), encoding="utf-8") as f:
    README = f.read()

# version 1
setup(name='multi_match',
      version='0.0.1',
      description='A package to detect chains.',
     author='Thomas Giacomo Nies',
      author_email='thomas.nies@uni-goettingen.de',
      long_description_content_type="text/markdown",
      long_description=README,
      license='MIT',
      packages=['multi_match'],
      include_package_data=True,
      install_requires=[
          'numpy',
          'ortools>=9.4',
          'scipy',
          'matplotlib',
           'scikit-image',
           'matplotlib_scalebar',
           'pandas'
      ],
      zip_safe=False)
