# EmoInt
EmoInt can used for [affective computing](https://en.wikipedia.org/wiki/Affective_computing)
like sentiment analysis, emotion classification, emotion intensity computing etc. This project is developed
during [WASSA 2017](http://optima.jrc.it/wassa2017/) Emotion Intensity Task. It is inspired by
[AffectiveTweets](https://github.com/felipebravom/AffectiveTweets) repo (baseline model for the Emotion Intensity Task).
This project project has scripts for ensemble creation. Finally we'll demo the sample application go live using
[Google Cloud](https://cloud.google.com/).

## Install
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
