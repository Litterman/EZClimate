from setuptools import setup, find_packages

try:
    from distutils.command.build_py import build_py_2to3 as build_py
except ImportError:
    from distutils.command.build_py import build_py
    
def readme():
    with open('README.md') as f:
        return f.read()

setup(name='ezclimate',
      version='1.0.1',
      description='EZ-Climate model',
      long_description=readme(),
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3, 2',
        'Topic :: Climat Change :: Pricing SCC',
      ],
      keywords='EZ-climate, social cost of carbon, SCC, climate pricing',
      url='http://github.com/Litterman/EZClimate',
      author='Robert Litterman, Kent Daniel, Gernot Wagner',
      license='MIT',
      packages=find_packages(),
      install_requires=['numpy',],
      include_package_data=False,
      zip_safe=True,
      cmdclass = {'build_py': build_py},
      use_2to3=True
      )

