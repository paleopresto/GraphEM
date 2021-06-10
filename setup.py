from setuptools import find_packages
from distutils.core import setup, Extension
from distutils import sysconfig
from Cython.Distutils import build_ext
import os
__version__ = '0.1.2'

with open('README.rst', 'r') as fh:
    long_description = fh.read()

quiclib = Extension('quiclib', sources=['./GraphEM/QUIC.cpp'])

class NoSuffixBuilder(build_ext):
    def get_ext_filename(self, ext_name):
        filename = super().get_ext_filename(ext_name)
        suffix = sysconfig.get_config_var('EXT_SUFFIX')
        ext = os.path.splitext(filename)[1]
        return os.path.join('./GraphEM', filename.replace(suffix, "")+ext)


setup(
    name='GraphEM',  # required
    version=__version__,
    description='Gaussian Markov random ﬁelds embedded within an EM algorithm',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    author='Feng Zhu, Dominique Guillot, Julien Emile-Geay',
    author_email='fengzhu@usc.edu, dguillot@udel.edu, julieneg@usc.edu',
    url='https://github.com/fzhu2e/GraphEM',
    packages=find_packages(),
    include_package_data=True,
    license='GPL-3.0 license',
    zip_safe=False,
    cmdclass={'build_ext': NoSuffixBuilder},
    ext_modules=[quiclib],
    keywords='GraphEM',
    classifiers=[
        'Natural Language :: English',
        'Programming Language :: Python :: 3.8',
    ],
    install_requires=[
        'LMRt',
    ],
)
