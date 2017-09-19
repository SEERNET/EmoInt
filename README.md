# EmoInt
[![Travis Build Status](https://api.travis-ci.org/SEERNET/EmoInt.svg)](https://travis-ci.org/SEERNET/EmoInt)
[![CircleCI Build Status](https://circleci.com/gh/SEERNET/EmoInt.svg?style=svg)](https://circleci.com/gh/SEERNET/EmoInt)
[![Coverage Status](https://coveralls.io/repos/github/SEERNET/EmoInt/badge.svg)](https://coveralls.io/github/SEERNET/EmoInt)

EmoInt can be used for [affective computing](https://en.wikipedia.org/wiki/Affective_computing)
like sentiment analysis, emotion classification, emotion intensity computing etc. This project is developed
during [WASSA 2017](http://optima.jrc.it/wassa2017/) Emotion Intensity Task. It is inspired by
[AffectiveTweets](https://github.com/felipebravom/AffectiveTweets) repo (baseline model for the Emotion Intensity Task).
This project contains high level wrappers for combining various word embeddings and scripts for creating ensembles. 

## Install

### Pre-requisites
Some word-embeddings need to be downloaded separately for using all available featurizers.

Note: Instructions to access these resources can be found [here](http://saifmohammad.com/WebPages/AccessResource.htm)

The relevant word embeddings are:

* NRC Affect Intensity: [Link](http://saifmohammad.com/WebPages/AffectIntensity.htm). Download to `emoint/resources/nrc_affect_intensity.txt.gz`
* NRC Emotion Wordlevel Lexicons: [Link](http://saifmohammad.com/WebPages/lexicons.html). Download to `emoint/resources/NRC-emotion-lexicon-wordlevel-v0.92.txt.gz`
* Sentiment140: [Link](http://saifmohammad.com/WebPages/lexicons.html). Download to `emoint/resources/Sentiment140-Lexicon-v0.1` 

### Reformatting
The NRC Emotion Wordlevel Lexicons are not in the standard format, we've provided a script to reformat it in the required format.

```python

python emoint/utils/reformat.py emoint/resources/NRC-emotion-lexicon-wordlevel-v0.92.txt.gz

```

### Installing

The package can be installed as follows:
```
python setup.py install
```

## Usage
You can learn how to use the featurizers by following these notebooks in [examples](emoint/examples) directory
 1. [Cornell Movie Review](http://www.cs.cornell.edu/people/pabo/movie-review-data/) -- [MovieReview.ipynb](emoint/examples/MovieReview.ipynb)
 2. [WASSA 2017 Emotion Intensity](http://optima.jrc.it/wassa2017/) -- [EmotionIntensity.ipynb](emoint/examples/EmotionIntensity.ipynb)


## Running Tests
```
python -m unittest discover -v
```

## Maintainers
- [Venkatesh Duppada](venkatesh-1729.github.io)
- [Sushant Hiray](sushant-hiray.github.io)
