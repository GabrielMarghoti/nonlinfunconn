# Nonlinear Functional Connectivity

## Overview
This repository provides implementations for analyzing nonlinear functional connectivity in complex systems. It includes methods for:
- **Convolution kernel fitting** to model interactions in complex networks based on measured data.
- **Nonequilibrium Green Functions (NEGF)** for studying signal propagation in nonlinear dynamical models.

Our approach allows kernel fitting in any complex network. Specifically, we apply it to the mapped atlas of *C. elegans*, enabling a deeper understanding of its neural connectivity.

## Features
- **Convolution Kernel Fitting**: Models the response of complex systems using data-driven approaches; such kernels might encompass linear or nonlinear assumptions. In the nonlinear case, we develop non-translational kernels based on NonEquilibrium Green Functions (NEGF).
- **Nonequilibrium Green Functions**: Computes signal transmission modulator dynamics in networked systems.
- **Application to *C. elegans***: Utilized for analyzing the *C. elegans* brain atlas.

## Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/gabrielmarghoti/nonlinfunconn.git
cd nonlinfunconn
sudo python3 setup.py install
```

## Usage

### Nonequilibrium Green Functions
```python
TODO
```

### Data for *C. elegans* Brain Analysis
The data used for analyzing *C. elegans* neural networks is available at: [OSF Repository](https://osf.io/e2syt/)

## References
For more details on the *C. elegans* brain analysis approach, refer to the paper: [Nature](https://www.nature.com/articles/s41586-023-06683-4)

## Dependencies
- Python 3.8+
- NumPy
- SciPy
- NetworkX
### For C. Elegans analysis:
- leiferlab/pumpprobe
- leiferlab/wormdatamodel
- leiferlab/wormbrain

  
## Contact
For questions, please contact [gabrielmarghoti@gmail.com] or open an issue.

