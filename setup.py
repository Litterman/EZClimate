from setuptools import setup, find_packages

try:
    from pypandoc import convert
except ImportError:
    import codecs
    read_md = lambda f: codecs.open(f, 'r', 'utf-8').read()
else:
    read_md = lambda f: convert(f, 'rst')

setup(name='ezclimate',
      version='2.0.4',
      description='EZ-Climate model',
      long_description=read_md('README.md'),
      classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7'
      ],
      keywords='EZ-climate, optimal carbon price, CO2 tax, social cost of carbon, SCC, social cost of carbon dioxide, SC-CO2',
      url='http://github.com/Litterman/EZClimate',
      author='Gernot Wagner, Kent Daniel, Robert Litterman',
      license='MIT',
      packages=find_packages(),
      install_requires=['numpy',],
      include_package_data=False,
      zip_safe=True
      )
