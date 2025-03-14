# NewsSum

A neural news summarization tool. Collect google news topics, crawl related news articles, generate summaries.
Including the following units:
1. BasicSum: traditional freq-based news-summary generator.
2. LSTM-Attetion: Neural network model that summarizes news.
3. WikiNews Dataset: A list of news articles crawled from WikiNews, every element(event) of the list contains at least three articles from mainstream website talking about the event.

## Run
python3 main.py

## Demo

http://news.qin.ee

## Pretrained Model Download

https://drive.google.com/file/d/0BwzJqD-PyhaoTVhZZVR2UTN6TU0/view?usp=sharing

## Data

https://docs.google.com/uc?id=0B0Obe9L1qtsnSXZEd0JCenIyejg&export=download

## Dependency

NLTK

numpy

Tensorflow

Newspaper

## Acknowledgement

[GNP](https://github.com/mPAND/gnp) 

[NeuralSum](https://github.com/cheng6076/NeuralSum)
For their dataset and dataset reader and LSTM architecture.

## License
MIT
