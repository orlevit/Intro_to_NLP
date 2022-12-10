import os
import argparse
import numpy as np
from time import time
from collections import defaultdict

BASE_LOC = r'/home/or/dev/Intro_to_NLP/task2'
POS_DATA_LOC = os.path.join(BASE_LOC, r'data/pos')
POS_TRAIN_FILE = os.path.join(POS_DATA_LOC, 'ass1-tagger-train')
POS_DEV_FILE = os.path.join(POS_DATA_LOC, 'ass1-tagger-dev')
#POS_TEST_FILE = os.path.join(POS_DATA_LOC, 'ass1-tagger-test-input')

NONE_EXIST = -1
NONE_POS_IND = 0
LEFT_POS_IND = 1
RIGHT_POS_IND = 2
BOTH_POS_IND = 3

def read_file(location):
    file = open(location, "r")
    file_lines = file.readlines()
    file.close()
    return file_lines

def possible_location(annotated_location):
    len_line = len(annotated_location)
    dist_tokens = np.ones(len_line) * -1

    for loc in range(len_line):
        if annotated_location[loc] == 0:
            loc_value = 0
            if loc + 1 < len_line:
                if annotated_location[loc + 1] == 1:
                    loc_value += 2

            if loc - 1 >= 0:
                if annotated_location[loc -1] == 1:
                    loc_value += 1

            dist_tokens[loc] = loc_value
    print(dist_tokens)
    return dist_tokens

def split_to_lists(line):
    words_list =[]
    pos_list = []
    
    for token_and_pos in line.split():
        token, pos = token_and_pos.rsplit('/',1)
        words_list.append(token)
        pos_list.append(pos)
        
    return words_list, pos_list

def match_value(the_dict, indication, right_pos, left_pos, match_any):
    try:
        gotten_pos_val = NONE_EXIST
        if indication == BOTH_POS_IND:
            if right_pos in the_dict['right_pos'] and \
            left_pos in the_dict['right_pos'][right_pos]['left_pos']:
                gotten_pos_val = the_dict['right_pos'][right_pos]['left_pos'][left_pos]['left_count']
                return gotten_pos_val
            
        if indication == RIGHT_POS_IND or match_any:
            if right_pos in the_dict['right_pos']:
                gotten_pos_val = the_dict['right_pos'][right_pos]['right_count']
                return gotten_pos_val
            
        if indication == LEFT_POS_IND or match_any:
            if left_pos in the_dict['left_pos']:
                gotten_pos_val = the_dict['left_pos'][left_pos]['left_count']
                return gotten_pos_val
            
        if indication == NONE_POS_IND or match_any:
            gotten_pos_val = the_dict['wp_count']
            return gotten_pos_val
    except:
        import pdb; pdb.set_trace();
    return gotten_pos_val

def best_pos_in_loc(loc, the_dict, words_list, pos_list, indication, match_any):
    best_pos = ''
    best_count = NONE_EXIST
    
    if words_list[loc] in the_dict.keys():

        left_pos = None
        right_pos = None  
        
        if loc > 0:
            left_pos = pos_list[loc - 1]

        if loc + 1 < len(pos_list):
                right_pos = pos_list[loc + 1]
        
#     try:
        for pos, pos_values in the_dict[words_list[loc]].items():
            val = match_value(pos_values, indication, right_pos, left_pos, match_any)
            if best_count < val:
                best_count = val
                best_pos = pos
#     except Exception as e: 
#         print(e)

#         import pdb; pdb.set_trace();
    return best_count, best_pos

def get_pos_for_indication(all_possible_locs, indication, the_dict, words_list, pos_list, match_any):
    specific_possible_locs = np.where(all_possible_locs == indication)[0]
#     try:
    best_pos = ''
    best_count = NONE_EXIST
    best_loc = NONE_EXIST

    for loc in specific_possible_locs:
        tested_val, tested_pos = best_pos_in_loc(loc, the_dict, words_list, pos_list, indication, match_any)
        if tested_pos != NONE_EXIST:
            if  best_count < tested_val:
                best_count = tested_val
                best_pos = tested_pos
                best_loc = loc
#     except:
#         import pdb; pdb.set_trace();                
    return best_count, best_pos, best_loc

def get_best_pos(file_lines, annotated_location, words_list, pos_list, the_dict, match_any):
    possible_locs = possible_location(annotated_location)

    best_count, best_pos, loc = get_pos_for_indication(possible_locs, BOTH_POS_IND, the_dict, words_list, pos_list, match_any)
    if best_count != NONE_EXIST:
        return best_pos, loc
    best_count, best_pos, loc = get_pos_for_indication(possible_locs, RIGHT_POS_IND, the_dict, words_list, pos_list, match_any)
    if best_count != NONE_EXIST:
        return best_pos, loc
    best_count, best_pos, loc = get_pos_for_indication(possible_locs, LEFT_POS_IND, the_dict, words_list, pos_list, match_any)
    if best_count != NONE_EXIST:
        return best_pos, loc
    best_count, best_pos, loc = get_pos_for_indication(possible_locs, NONE_POS_IND, the_dict, words_list, pos_list, match_any)
    return best_pos, loc


import json
with open(os.path.join(BASE_LOC, r'dict_temp.json'), "r") as json_file:
    words_pos_dict = json.load(json_file)    
    
file_lines = read_file(POS_DEV_FILE)

# none exsiting ?????????????????????
lines_output = []
matching_pos = 0
total_pos = 0
for ii,file_line in enumerate(file_lines):
    match_any = False
    splitted_line = file_line.split() 
    len_line = len(splitted_line)
    annotated_location = np.zeros(len_line)
    annotated_token = [''] * len_line
    words_list, label_pos = split_to_lists(file_line)
    print('----->',ii)
    while not all(annotated_location):
#         print('--------------------')
        import pdb; pdb.set_trace();                

        best_pos, loc = get_best_pos(file_lines, annotated_location, words_list, annotated_token, words_pos_dict, match_any)
#         print('--------------------')
        if match_any and (loc == NONE_EXIST):
            match_

        if loc == NONE_EXIST:
            match_any = True
        else:
            annotated_location[loc] = 1
            annotated_token[loc] = best_pos

        print(loc)
    lines_output.append([f'{i}/{j}' for i,j in zip(words_list,annotated_token)])
    matching_pos += sum(np.asarray(annotated_token)==np.asarray(label_pos))
    total_pos += len_line
import pdb; pdb.set_trace();                

    
print(f'The accuracy over the data set is {matching_pos/total_pos}')

        
        
        
        