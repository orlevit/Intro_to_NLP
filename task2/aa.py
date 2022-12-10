import os
import argparse
import numpy as np
from time import time
from pprint import pprint
from collections import defaultdict

BASE_LOC = r'/home/or/dev/Intro_to_NLP/task2'
POS_DATA_LOC = os.path.join(BASE_LOC, r'data/pos')
POS_TRAIN_FILE =os.path.join(POS_DATA_LOC, 'ass1-tagger-train')

def add_wp_to_dict(the_dict, word, pos):
    try:
        if word in the_dict:
            if pos in the_dict[word]:
                the_dict[word][pos]['wp_count'] += 1
            else:
                the_dict[word][pos] = {'wp_count':1, 'right_pos':{}, 'left_pos':{}}

        else:
            the_dict[word] = {pos: {'wp_count':1, 'right_pos':{}, 'left_pos':{}}}
    except:
        import pdb; pdb.set_trace();

def add_right_pos_to_dict(the_dict, word, pos, right_pos):
    try:
        #import pdb; pdb.set_trace();
        if right_pos in the_dict[word][pos]['right_pos']:
            the_dict[word][pos]['right_pos'][right_pos]['right_count'] += 1
        else:
            the_dict[word][pos]['right_pos'][right_pos] = {'right_count':1, 'left_pos':{}}
    except:
        import pdb; pdb.set_trace();

def add_left_pos_to_dict(the_dict, word, pos, left_pos):
    if left_pos in the_dict[word][pos]['left_pos']:
        the_dict[word][pos]['left_pos'][left_pos]['left_count'] += 1
    else:
        the_dict[word][pos]['left_pos'][left_pos] = {'left_count':1}
        
def add_wp_right_pos_left_pos_to_dict(the_dict, word, pos, right_pos, left_pos):
    try:    
        if left_pos in the_dict[word][pos]['right_pos'][right_pos]['left_pos']:
            the_dict[word][pos]['right_pos'][right_pos]['left_pos'][left_pos]['left_count'] += 1
        else:
            the_dict[word][pos]['right_pos'][right_pos]['left_pos'][left_pos] = {'left_count':1} 
    except:
        import pdb; pdb.set_trace();        
def split_token_pos(token_and_pos, loc):
    try:    
        token, pos = token_and_pos[loc].rsplit('/',1)
    except:
        import pdb; pdb.set_trace()
    return token, pos
    
file = open(POS_TRAIN_FILE, "r")
sentence = []
sentences = []
file_lines = file.readlines()
file.close()

words_pos_dict = defaultdict()
for file_line in file_lines:
    splitted_line = file_line.split()
    len_line = len(splitted_line)
    for ii, _ in enumerate(splitted_line):
           # print(ii, len_line,splitted_line)
        token, pos = split_token_pos(splitted_line, ii)
        add_wp_to_dict(words_pos_dict, token, pos)
        if ii  + 1 < len_line:
            right_token, right_pos = split_token_pos(splitted_line, ii + 1)
            add_right_pos_to_dict(words_pos_dict, token, pos, right_pos)
        if ii - 1 <= 0:
            left_token, left_pos = split_token_pos(splitted_line, ii - 1)
            add_left_pos_to_dict(words_pos_dict, token, pos, left_pos)
        if (ii - 1 <= 0) and (ii  + 1 < len_line):
            add_wp_right_pos_left_pos_to_dict(words_pos_dict, token, pos, right_pos, left_pos)
        #break
import pdb; pdb.set_trace();
