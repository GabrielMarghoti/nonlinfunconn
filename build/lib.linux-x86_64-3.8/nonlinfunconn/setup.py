from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import os

class CustomBuildExtCommand(build_ext):
    """Custom build_ext command to include numpy headers dynamically."""
    def build_extensions(self):
        import numpy as np
        for ext in self.extensions:
            ext.include_dirs.append(np.get_include())  # Append numpy headers
        super().build_extensions()

# Define the C++ extensions (without hardcoded numpy include)

_convolution = Extension(
    name='utils._convolution',
    sources=[
        os.path.join(os.path.dirname(__file__), 'utils', '_convolution.cpp'),
        os.path.join(os.path.dirname(__file__), 'utils', 'convolution.cpp')
    ],
    extra_compile_args=['-O3']  # Optimization flag
)

_integration = Extension(
    name='utils._integration',
    sources=[os.path.join(os.path.dirname(__file__), 'utils', '_integration.cpp')],
    extra_compile_args=['-O3']  # Optimization flag
)

setup(
    name='nonlinfunconn',
    version='0.1',
    packages=find_packages(),  # Automatically find packages
    install_requires=[
        'numpy',
        'scipy',
    ],
    ext_modules=[_integration, _convolution],  # List of C++ extensions
    cmdclass={'build_ext': CustomBuildExtCommand},  # Use the custom build_ext command
    author='Gabriel Marghoti',
    author_email='gabrielmarghoti@gmail.com',
    description='Convolution kernel fitting for complex systems dynamics.',
    url='https://github.com/GabrielMarghoti/nonlinfunconn',
)
