# EZ-Climate

EZ-Climate is a model for pricing carbon dioxide emission. It explores the implications of these richer preference specifications for the optimal ![equation](http://latex.codecogs.com/gif.latex?CO_2) price. We develop the EZ-Climate model, a simple discrete-time model in which the representative agent has an Epstein-Zin preference specification, and in which uncertainty about the effect of ![equation](http://latex.codecogs.com/gif.latex?CO_2) emissions on global temperature and on eventual damages is gradually resolved over time. In the EZ-Climate model the optimal price is equal to the price of one ton of ![equation](http://latex.codecogs.com/gif.latex?CO_2) emitted at any given point in time that maximizes the utility of the representative agent at that time. We embed a number of features including tail risk, the potential for technological change, and backstop technologies. In contrast to most modeled carbon price paths, the EZ-Climate model suggests a high optimal carbon price today that is expected to decline over time. It also points to the importance of backstop technologies and to potentially very large.

## Downloads

You can find the most recent releases at: https://pypi.python.org/pypi/ezclimate/.

## Documentation
See the [EZ-Climate User's Guide](https://oscarsjogren.github.io/dlw/) for EZ-Climate documentation.

In order to get the tip documentation, change directory to the `docs` subfolder and type in `make html`, the documentation will be under `../../ez_climate_docs/html`. You will need [Sphinx](http://sphinx.pocoo.org) to build the documentation.

See [Applying Asset Pricing Theory to Calibrate the Price of Climate Risk](https://gwagner.com/daniel-litterman-wagner-applying-asset-pricing-theory-to-calibrate-the-price-of-climate-risk/) for the latest working paper employing this code.

## Installation

We encourage you to use pip to install ezclimate on your system. 

```bash
pip install ezclimate
```

If you wish to build from sources, download or clone the repository.

```bash
python setup.py install
```

## Requirements

EZ-Climate is compatible with Python 2 and 3. [Numpy](http://www.numpy.org/) is required, and we recommend [matplotlib](http://www.matplotlib.org/) for visualization of results.

## Authors

* Robert Litterman
* Kent Daniel
* Gernot Wagner

