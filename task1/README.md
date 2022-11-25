# Intro_to_NLP
Install requirements:
```
pip install -r requirements.txt
```
Download the hebrew wikipedia corpus whichis available here: https://u.cs.biu.ac.il/~yogo/hebwiki/

Use the “parsed sentences” file. It is compressed. On linux, you can open it with the
commandline command:
```
gunzip wikipedia.deps.gz
```

**Part 1:**
Goal: 
Compute distributional similarities for words based on corpus data and to assess and evaluate those similarities. 

Basic usage:
```
python prog.py -io WIKI_FILE_LOCATION -ic CONVERTED_WIKI_FILE_LOCATION 
```
Example:
WIKI_FILE_LOCATION - /home/or/NLP/wikipedia.deps
CONVERTED_WIKI_FILE_LOCATION - /home/or/NLP/wiki_converted.pkl

For more options, run;
```
python prog.py -h


output:
usage: prog.py [-h] [-io INPUT_ORIGINAL] [-ic INPUT_CONVERTED]
               [-ci CONVERT_INDICATION] [-tw TOP_WORDS_FILE]
               [-tf TOP_FEATURES_FILE] [-cw COMMON_WORDS_FILE]
               [-cf COMMON_FEATURES_FILE] [-tv TOP_VALUES] [-tc TOP_COUNTS]

optional arguments:
  -h, --help            show this help message and exit
  -io INPUT_ORIGINAL, --input-original INPUT_ORIGINAL
                        The originl wiki file
  -ic INPUT_CONVERTED, --input-converted INPUT_CONVERTED
                        Converted wiki file location(only required columns out
                        of the file)
  -ci CONVERT_INDICATION, --convert-indication CONVERT_INDICATION
                        Whether the converted file already exist. No
                        conversonis needed
  -tw TOP_WORDS_FILE, --top-words-file TOP_WORDS_FILE
                        The location of the top wordsfile
  -tf TOP_FEATURES_FILE, --top-features-file TOP_FEATURES_FILE
                        The location of the top features file
  -cw COMMON_WORDS_FILE, --common-words-file COMMON_WORDS_FILE
                        The location of the common word file
  -cf COMMON_FEATURES_FILE, --common-features-file COMMON_FEATURES_FILE
                        The location of the common features file
  -tv TOP_VALUES, --top-values TOP_VALUES
                        Top X values in features/similarities
  -tc TOP_COUNTS, --top-counts TOP_COUNTS
                        Top X common features/similarities
```

**Part 2:**
Goal: 
Compute similarities using pre-trained word2vec word embeddings and perform various queries on these similarities.

Usage:
Run each excercise in different jupyter notebook. 
In the begining of each notebook, there are instalation, so they can be ran without the requirement installation.
