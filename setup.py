from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='EmoInt',
    version='0.0.1',
    description='Affective Computing',
    long_description=long_description,
    url='https://github.com/seernet/EmoInt',
    author='Venkatesh Duppada',
    author_email='venkatesh.duppada@seernet.io',
    # Todo - Choose license
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        # Todo - Choose license
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        # Todo - Make it python 3 compatible ?
        'Programming Language :: Python :: 3',
    ],
    # Todo - Add more key words ?
    keywords='sentiment emotion',
    packages=find_packages(),
    install_requires=[],
    extras_require={},
    package_data={
        'emoint': ['resources/*', 'resources/NRC-Hashtag-Sentiment-Lexicon-v0.1/*', 'resources/SentiStrength/*',
                   'resources/Sentiment140-Lexicon-v0.1/*', 'resources/emoint/*']
    },
    data_files=[],
    entry_points={},
)
