# Intro_to_NLP
Python version: 3.7.9

Install requirements:
```
pip install -r requirements.txt
```
Download and unzip the files:
1) Part Of Speech data - "pos.tgz".
2) Name Entities Recognition - "ner.tgz".


**Part 1:**

Given a text, assign each word in the text its correct part-of-speech based on the out-of-the-box vectors, without
training any classifiers.

1.1
Goal: 
Using only word count.

Usage:
```
python create_pos_1.py -itr INPUT_TRAIN -id INPUT_DEV -it INPUT_TEST -ot OUTPUT_TEST
```

1.2
Goal: 
Using word count and static word vectors.


Usage:
```
python create_pos_1.py -itr INPUT_TRAIN -id INPUT_DEV -it INPUT_TEST -ot OUTPUT_TEST
```

1.3
Goal:
Using word count and static/Contextualized word vectors.


Usage:
```
python create_pos_1.py -itr INPUT_TRAIN -id INPUT_DEV -it INPUT_TEST -ot OUTPUT_TEST
```
**Part 2:**

Goal: 
perform named entity recognition

Usage:
```
python create_pos_1.py -itr INPUT_TRAIN -id INPUT_DEV -it INPUT_TEST -od OUTPUT_DEV -ot OUTPUT_TEST
```
