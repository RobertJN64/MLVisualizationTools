from setuptools import setup
import os


include = []
for item in list(os.walk('MLVisualizationTools')):
    if 'pycache' not in item[0]:
        include.append(item[0].replace('\\', '.'))


setup(
    name='MLVisualizationTools',
    url='https://github.com/RobertJN64/MLVisualizationTools',
    author='Robert Nies',
    author_email='robertjnies@gamil.com',
    # Needed to actually package something
    packages=include,
    install_requires=['pandas'],
    extras_require={'dash': ['dash', 'plotly', 'dash_bootstrap_components>=1.0.0'],
                    'dash-notebook': ['dash', 'plotly', 'dash_bootstrap_components>=1.0.0', 'jupyter-dash']},
    # *strongly* suggested for sharing
    version='0.0.3',
    # The license can be anything you like
    license='MIT',
    description='A set of functions and demos to make machine learning projects easier to understand.',
    long_description=open('README.md').read(),
)