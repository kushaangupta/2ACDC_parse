from setuptools import setup, find_packages

setup(
    name='linear2ac',
    version='0.0.1',
    author='Johan Winnubst',
    author_email='johan@e11.bio',
    packages=find_packages(include=['linear2ac', 'linear2ac.*']),  # Include all submodules
    install_requires=['umap-learn', 'fpdf2', 'plotly', 'figrid'],
    zip_safe=False
)
