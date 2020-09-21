from setuptools import setup, find_packages

setup(name='MIDASpy',
	  packages=['MIDASpy'],
      version= '1.0.2',
      licence= 'Apache',
      description= 'Implementation of Multiple Imputation using Denoising Autoencoders',
      url= 'http://github.com/ranjitlall/MIDAS',
      author= 'Ranjit Lall, Alex Stenlake, and Thomas Robinson',
      author_email= 'R.Lall@lse.ac.uk',
      download_url = 'https://github.com/MIDASverse/MIDASpy/releases/download/v1.0.1/MIDASpy-1.0.1.tar.gz',
      install_requires = [
        'tensorflow>=1.10',
        'numpy>=1.5',
        'scikit-learn',
        'matplotlib',
        'pandas>=0.19',
      ],
      keywords = ['multiple imputation','neural networks','tensorflow'],
      classifiers = [
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: Apache Software License',   # Again, pick a license
        'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        ],
      )
