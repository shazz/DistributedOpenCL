from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='sample',
    version='0.1.0',
    description='RPyOpenCL',
    long_description=readme,
    author='Shazz',
    author_email='shazz@metaverse.fr',
    url='https://github.com/shazz/DistributedOpenCL',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)
