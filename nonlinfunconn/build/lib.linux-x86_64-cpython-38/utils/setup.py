from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import numpy as np

class CustomBuildExtCommand(build_ext):
    """Custom build_ext command to include numpy headers during the build process."""

    def run(self):
        # Import numpy here to ensure it is available during the build process
        import numpy as np
        
        # Add numpy headers to include_dirs
        self.include_dirs.append(np.get_include())
        
        # Call the original build_ext command
        super().run()

# Define the C++ extensions
_integration = Extension(
    name='_integration',
    sources=['_integration.cpp'],
    include_dirs=[np.get_include()],  # Include numpy headers
    extra_compile_args=['-O3']  # Optimization flag
)

_convolution = Extension(
    name='_convolution',
    sources=['_convolution.cpp', 'utils/convolution.cpp'],
    include_dirs=[np.get_include()],  # Include numpy headers
    extra_compile_args=['-O3']  # Optimization flag
)

setup(
    name='NonlinearFunctionalConnectivity',
    version='0.1',
    packages=find_packages(),  # Automatically find packages in the directory
    install_requires=[
        'numpy',
        'scipy',
    ],
    ext_modules=[_integration, _convolution],  # List of C++ extensions
    cmdclass={
        'build_ext': CustomBuildExtCommand,  # Use the custom build_ext command
    },
    author='Gabriel Marghoti',
    author_email='gabrielmarghoti@gmail.com',
    description='Convolution kernel fitting for complex systems dynamics.',
    url='https://github.com/GabrielMarghoti/NonlinearFunctionalConnectivity',
)