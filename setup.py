from setuptools import setup, find_packages

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(name='MIDASpy',
	  packages=['MIDASpy'],
      version= '1.2.3',
      licence= 'Apache',
      description= 'Multiple Imputation with Denoising Autoencoders',
      long_description_content_type='text/markdown',
      long_description=long_description,
      url= 'http://github.com/MIDASverse/MIDASpy',
      author= 'Ranjit Lall, Alex Stenlake, and Thomas Robinson',
      author_email= 'R.Lall@lse.ac.uk',
      download_url = 'https://github.com/MIDASverse/MIDASpy/archive/v1.2.3.tar.gz',
      install_requires = [
        'tensorflow>=1.10; sys_platform != "darwin" or platform_machine != "arm64"',
        'tensorflow-macos>=1.10; sys_platform == "darwin" and platform_machine == "arm64"',
        'numpy>=1.5',
        'scikit-learn',
        'matplotlib',
        'pandas>=0.19',
        'tensorflow_addons>=0.11',
        'statsmodels',
        'scipy'
      ],
      keywords = ['multiple imputation','neural networks','tensorflow'],
      classifiers = [
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: Apache Software License',   # Again, pick a license
        'Programming Language :: Python :: 3',      #Specify which python versions that you want to support
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        ],
      )
