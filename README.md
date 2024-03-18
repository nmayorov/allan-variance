# allan_variance

Simple Python package to compute Allan variance and estimate noise parameters from it.
It also provides a power-law spectrum noise generator for simulation purposes.
The application in mind was analysis of random noise in inertial sensors.

## Installation

### Installing from pypi

```shell
pip install allan-variance
```

### Installing from source

To perform a regular install, execute in the cloned repository directory: 
```shell
pip install .
```
To perform an editable (inplace) install:
```shell
pip install -e .
```

## Dependencies

Runtime dependencies include (versions in parentheses were used for the latest development):

* numpy (1.25.2)
* scipy (1.11.3)
* pandas (2.1.1)

## Examples

An [example](https://github.com/nmayorov/allan-variance/blob/master/example.ipynb) included in 
the repository demonstrates usage.
