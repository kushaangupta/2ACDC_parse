from setuptools import setup
setup(
    name = 'linear2ac',
    version = '0.0.1',
    author = 'Johan Winnubst',
    author_email = 'johan@e11.bio',
    packages = ['linear2ac'],
    install_requires=['umap-learn','fpdf2','plotly','figrid'],
    zip_safe = False
)