from setuptools import setup, find_packages

setup(name='midas',
      version= '1.0',
      description= 'Implementation of Multiple Imputation using Denoising Autoencoders',
      url= 'http://github.com/ranjitlall/MIDAS',
      author= 'Ranjit Lall, Alex Stenlake, and Thomas Robinson',
      author_email= 'R.Lall@lse.ac.uk',
      licence= 'Apache',
      #packages= find_packages(),
      packages=['midas'],
      zip_safe=False)
