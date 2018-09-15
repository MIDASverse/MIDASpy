from setuptools import setup, find_packages

setup(name='midas',
      version= '0.2',
      description= 'Implementation of Multiple Imputation using Denoising Autoencoders',
      url= 'http://github.com/Oracen/MIDAS',
      author= 'Alex Stenlake',
      author_email= 'alex.stenlake@gmail.com',
      licence= 'Apache',
      packages= find_packages(),
      zip_safe=False)