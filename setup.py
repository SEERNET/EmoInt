from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='EmoInt',
    version='0.0.2',
    description='Affective Computing',
    long_description=long_description,
    url='https://github.com/seernet/EmoInt',
    author='Venkatesh Duppada',
    author_email='venkatesh.duppada@seernet.io',
    license='GPLv3',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
    ],
    keywords='sentiment emotion affective computing machine learning',
    packages=find_packages(),
    setup_requires=[
        # Setuptools 18.0 properly handles Cython extensions.
        'setuptools>=18.0',
        'cython',
    ],
    install_requires=[],
    extras_require={},
    package_data={
        'emoint': ['resources/*', 'resources/NRC-Hashtag-Sentiment-Lexicon-v0.1/*', 'resources/SentiStrength/*',
                   'resources/Sentiment140-Lexicon-v0.1/*', 'resources/emoint/*']
    },
    data_files=[],
    entry_points={},
)
