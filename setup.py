# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Samson Wang
# --------------------------------------------------------

from __future__ import print_function

from setuptools import Extension
from setuptools import setup
from distutils.command.build import build as _build
import os

# ref from https://stackoverflow.com/questions/54117786/add-numpy-get-include-argument-to-setuptools-without-preinstalled-numpy
class build(_build):
    def finalize_options(self):
        super().finalize_options()
        import builtins
        builtins.__NUMPY_SETUP__ = False
        import numpy as np
        # Obtain the numpy include directory.  This logic works across numpy versions.
        extension = next(m for m in self.distribution.ext_modules if m.name=='fuzzy_cython_bbox')
        try:
            extension.include_dirs.append(np.get_include())
        except AttributeError:
            extension.include_dirs.append(np.get_numpy_include())

with open("README.md", "r") as fh:
    long_description = fh.read()

if os.name == 'nt':
    compile_args = {'gcc': ['/Qstd=c99']}
else:
    compile_args = ['-Wno-cpp']

ext_modules = [
    Extension(
        name='fuzzy_cython_bbox',
        sources=['src/fuzzy_cython_bbox.pyx'],
        extra_compile_args = compile_args,
    )
]

setup(
    name='fuzzy_cython_bbox',
    setup_requires=["setuptools>=18.0","Cython","numpy"],
    install_requires=["Cython","numpy"],
    ext_modules=ext_modules,
    cmdclass={'build': build},
    version = '0.1.4',
    description = 'Standalone fuzzy_cython_bbox',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author = 'Edgardo Solano-Carrillo',
    author_email = 'Edgardo.SolanoCarrillo@dlr.de',
    url = 'https://github.com/DLR-MI/fuzzy_cython_bbox.git',
    keywords = ['fuzzy_cython_bbox']
)

