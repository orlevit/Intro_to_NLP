import re
import os
import argparse
import settings
import numpy as np
from tqdm import tqdm
from time import time
from pprint import pprint
from create_dictionary import *

BASE_LOC = r'/home/or/dev/Intro_to_NLP/task3/part5'
POS_DATA_LOC = os.path.join(BASE_LOC, r'data')
INPUT_FILE = os.path.join(POS_DATA_LOC, 'ass1-tagger-train')
OUTPUT_FILE = os.path.join(BASE_LOC, 'grammer_rules.txt')



def write_file(location, data):
    file = open(location, "w")
    for line in data:
        file.write(line)
        file.write('\n')
    file.close()


def split_to_lists(line, indication):
    words_list =[]
    pos_list = []
    
    if indication:
        for token_and_pos in line.split():
            token, pos = token_and_pos.rsplit('/',1)
            words_list.append(token)
            pos_list.append(pos)        

    else:
        for token in line.split():
            words_list.append(token)
        
    return words_list, pos_list

    
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default=INPUT_FILE, help='Input POS file')
    parser.add_argument('-o', '--output', type=str, default=OUTPUT_FILE, help='Output grammer rules')

    return parser.parse_args()


def create_grammer_rules(the_dict):
    grammer_rules = [] 
    for word, word_values in the_dict.items():
        for pos, pos_values in word_values.items():
            #print(pos_values)
            if re.match("^[A-Za-z0-9_-]*$", pos) and \
               re.match("^[A-Za-z0-9_-]*$", word):
               grammer_rule = str(pos_values['wp_count']) + '\t' + pos + '\t' + word
            grammer_rules.append(grammer_rule)
            if len(pos_values['right_pos']):
               for r_pos, r_pos_values in pos_values['right_pos'].items():
                   if len(r_pos_values['left_pos']):
                      for rl_pos, counts in r_pos_values['left_pos'].items():
                          if re.match("^[A-Za-z0-9_-]*$", pos) and \
                             re.match("^[A-Za-z0-9_-]*$", r_pos) and \
                             re.match("^[A-Za-z0-9_-]*$", rl_pos):

                             grammer_rule = str(counts['left_count']) + '\t' + pos + '\t' + r_pos + '\t' + rl_pos + '\t'
                             grammer_rules.append(grammer_rule)

    return grammer_rules
                     


def main(args):  
    # Create the dictionary
    tic = time()
    words_pos_dict = make_dictonary(args.input)
    toc = time()
    print('Ceate dictionary running time: ', round((toc- tic)/60, 2))
    
    # Report accuracy on dev set
    lines_output = create_grammer_rules(words_pos_dict)
    
    # Write to file
    write_file(args.output, lines_output)
    

if __name__ == '__main__':
    args = parse_arguments()
    main(args)
