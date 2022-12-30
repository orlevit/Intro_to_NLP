import re
import os
import argparse
import numpy as np
from tqdm import tqdm
from time import time
from pprint import pprint
from create_dictionary import *

BASE_LOC = r'/home/or/dev/Intro_to_NLP/task3/part5'
POS_DATA_LOC = os.path.join(BASE_LOC, r'data')
INPUT_FILE = os.path.join(POS_DATA_LOC, 'ass1-tagger-train')
OUTPUT_FILE = os.path.join(BASE_LOC, 'grammar5')



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
    parser.add_argument('-o', '--output', type=str, default=OUTPUT_FILE, help='Output grammar rules')

    return parser.parse_args()

def init_grammar_rules():
    grammar_rules = [] 
    grammar_rule = '1' + '\t' + 'ROOT' + '\t' + 'S' + ' .'
    grammar_rules.append(grammar_rule)
    grammar_rule = '1' + '\t' + 'ROOT' + '\t' + 'S' + ' !'
    grammar_rules.append(grammar_rule)
    grammar_rule = '1' + '\t' + 'ROOT' + '\t' + 'is it true that S' + ' ?'
    grammar_rules.append(grammar_rule)
    grammar_rule = '1' + '\t' + 'S' + '\t' + 'NP VP'
    grammar_rules.append(grammar_rule)
    grammar_rule = '1' + '\t' + 'VP' + '\t' + 'VB NP'
    grammar_rules.append(grammar_rule)
    grammar_rule = '1' + '\t' + 'NP' + '\t' + 'DT NN'
    grammar_rules.append(grammar_rule)
    grammar_rule = '0.1' + '\t' + 'NP' + '\t' + 'NP PP'
    grammar_rules.append(grammar_rule)
    grammar_rule = '1' + '\t' + 'PP' + '\t' + 'IN NP'
    grammar_rules.append(grammar_rule)
    grammar_rule = '1' + '\t' + 'NN' + '\t' + 'JJ NN'
    grammar_rules.append(grammar_rule)

    return grammar_rules

def create_grammar_rules(the_dict):
    grammar_rules = init_grammar_rules()
    for word, word_values in the_dict.items():
        for pos, pos_values in word_values.items():
            #print(pos_values)
            if re.match("^[A-Za-z0-9_-]*$", pos) and \
               re.match("^[A-Za-z0-9_-]*$", word):
               grammar_rule = str(pos_values['wp_count']) + '\t' + pos + '\t' + word
            grammar_rules.append(grammar_rule)
            if len(pos_values['right_pos']):
               for r_pos, r_pos_values in pos_values['right_pos'].items():
                   if len(r_pos_values['left_pos']):
                      for rl_pos, counts in r_pos_values['left_pos'].items():
                          if re.match("^[A-Za-z0-9_-]*$", pos) and \
                             re.match("^[A-Za-z0-9_-]*$", r_pos) and \
                             re.match("^[A-Za-z0-9_-]*$", rl_pos):

                             grammar_rule = str(counts['left_count']) + '\t' + pos + '\t' + r_pos + '\t' + rl_pos + '\t'
                             grammar_rules.append(grammar_rule)

    return grammar_rules
                     


def main(args):  
    # Create the dictionary
    tic = time()
    words_pos_dict = make_dictonary(args.input)
    toc = time()
    print('Ceate dictionary running time: ', round((toc- tic)/60, 2))
    
    # Report accuracy on dev set
    lines_output = create_grammar_rules(words_pos_dict)
    
    # Write to file
    write_file(args.output, lines_output)
    

if __name__ == '__main__':
    args = parse_arguments()
    main(args)
