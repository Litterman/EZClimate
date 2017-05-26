from setuptools import setup, find_packages

try:
    from pypandoc import convert
except ImportError:
    import codecs
    read_md = lambda f: codecs.open(f, 'r', 'utf-8').read()
else:
    read_md = lambda f: convert(f, 'rst')

setup(name='ezclimate',
      version='1.2.1b1',
      description='EZ-Climate model',
      long_description=read_md('README.md'),
      classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4'
      ],
      keywords='EZ-climate, social cost of carbon, SCC, climate pricing',
      url='http://github.com/Litterman/EZClimate',
      author='Robert Litterman, Kent Daniel, Gernot Wagner',
      license='MIT',
      packages=find_packages(),
      install_requires=['numpy',],
      include_package_data=False,
      zip_safe=True
      )

