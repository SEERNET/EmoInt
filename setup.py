from setuptools import setup, find_packages
from codecs import open
from os import path

try: # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError: # for pip <= 9.0.3
    from pip.req import parse_requirements

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

install_reqs = parse_requirements('requirements.txt', session='session')
reqs = [str(ir.req) for ir in install_reqs]

setup(
    name='EmoInt',
    version='0.1.0',
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
    ],
    install_requires=reqs,
    extras_require={},
    package_data={
        'emoint': ['resources/*', 'resources/NRC-Hashtag-Sentiment-Lexicon-v0.1/*', 'resources/SentiStrength/*',
                   'resources/Sentiment140-Lexicon-v0.1/*', 'resources/emoint/*']
    },
    data_files=[],
    entry_points={},
)
