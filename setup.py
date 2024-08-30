import io
import sys
from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

if sys.version_info < (3, 6):
    sys.exit("Sorry, Python 3.5 is no longer supported. Please use Python versions from 3.6 to 3.10")

install_requires = ['numpy>=1.5', 'scikit-learn', 'matplotlib', 'pandas>=0.19', 'tensorflow_addons<0.20', 'statsmodels', 'scipy']
if sys.version_info >= (3, 8) and sys.version_info < (3, 11):
    install_requires.append('tensorflow<2.12.0; sys_platform != "darwin" or platform_machine != "arm64"')
    install_requires.append('tensorflow-macos<2.12.0; sys_platform == "darwin" and platform_machine == "arm64"')
else:
    install_requires.append('tensorflow>=1.10; sys_platform != "darwin" or platform_machine != "arm64"')
    install_requires.append('tensorflow-macos>=1.10; sys_platform == "darwin" and platform_machine == "arm64"')

setup(
    name='MIDASpy',
    packages=['MIDASpy'],
    version='1.4.0',
    license='Apache',
    description='Multiple Imputation with Denoising Autoencoders',
    long_description_content_type='text/markdown',
    long_description=long_description,
    url='http://github.com/MIDASverse/MIDASpy',
    project_urls={
        'Method article': 'https://doi.org/10.1017/pan.2020.49',
        'Software article': 'https://doi.org/10.18637/jss.v107.i09',
        'Source': 'https://github.com/MIDASverse/MIDASpy',
        'Issues': 'https://github.com/MIDASverse/MIDASpy/issues',
    },
    author='Ranjit Lall, Alex Stenlake, and Thomas Robinson',
    author_email='R.Lall@lse.ac.uk',
    python_requires='>=3.6, <3.11',
    install_requires=install_requires,
    keywords=['multiple imputation', 'neural networks', 'tensorflow'],
    extras_require={'test': ['pytest','matplotlib']},

    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)
