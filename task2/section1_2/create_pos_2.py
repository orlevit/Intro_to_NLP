import os
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from time import time
from pprint import pprint
from num2words import num2words
from sklearn.metrics.pairwise import cosine_similarity
from create_dictionary2 import make_dictonary, read_file, get_vecs, model_vectors

BASE_LOC = r'/RG/rg-tal/orlev/study/bar_ilan/Intro_to_NLP/task2'
POS_DATA_LOC = os.path.join(BASE_LOC, r'data/pos')
POS_TRAIN_FILE = os.path.join(POS_DATA_LOC, 'ass1-tagger-train')
POS_DEV_FILE = os.path.join(POS_DATA_LOC, 'ass1-tagger-dev')
POS_TEST_FILE = os.path.join(POS_DATA_LOC, 'ass1-tagger-test-input')
OUTPUT_FILE = os.path.join(POS_DATA_LOC, 'POS_preds_1.txt')

NONE_EXIST = -1
NONE_POS_IND = 0
LEFT_POS_IND = 1
RIGHT_POS_IND = 2
BOTH_POS_IND = 3


def write_file(location, data):
    file = open(location, "w")
    for line in data:
        file.write(line)
        file.write('\n')
    file.close()

    
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
#     print(dist_tokens)
    return dist_tokens


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
        
def get_location_vec(indication, words_list, loc): 
    if indication == BOTH_POS_IND:
       vec_list = [get_vecs(words_list[loc - 1]), get_vecs(words_list[loc]), get_vecs(words_list[loc + 1])]
       loc_vec = np.mean(vec_list, axis=0)

    if indication == RIGHT_POS_IND:
       vec_list = [get_vecs(words_list[loc]), get_vecs(words_list[loc + 1])]
       loc_vec = np.mean(vec_list, axis=0)

    if indication == LEFT_POS_IND:
       vec_list = [get_vecs(words_list[loc - 1]), get_vecs(words_list[loc])]
       loc_vec = np.mean(vec_list, axis=0)

    if indication == RIGHT_POS_IND:
       vec_list = [get_vecs(words_list[loc]), get_vecs(words_list[loc + 1])]
       loc_vec = np.mean(vec_list, axis=0)

    if indication == NONE_POS_IND:
       loc_vec = get_vecs(words_list[loc])

    return loc_vec


def match_value(the_dict, indication, right_pos, left_pos, match_any, words_list, loc):
    loc_vec = NONE_EXIST
    gotten_pos_val = NONE_EXIST

    if indication == BOTH_POS_IND:
        if right_pos in the_dict['right_pos'] and \
        left_pos in the_dict['right_pos'][right_pos]['left_pos']:
            gotten_pos_val = the_dict['right_pos'][right_pos]['left_pos'][left_pos]['left_count']

            loc_vec = get_location_vec(BOTH_POS_IND, words_list, loc)
            return gotten_pos_val, loc_vec

    if indication == RIGHT_POS_IND or match_any:
        if right_pos in the_dict['right_pos']:
            gotten_pos_val = the_dict['right_pos'][right_pos]['right_count']
    
            loc_vec = get_location_vec(RIGHT_POS_IND, words_list, loc)
            return gotten_pos_val, loc_vec

    if indication == LEFT_POS_IND or match_any:
        if left_pos in the_dict['left_pos']:
            gotten_pos_val = the_dict['left_pos'][left_pos]['left_count']

            loc_vec = get_location_vec(LEFT_POS_IND, words_list, loc)
            return gotten_pos_val, loc_vec

    if indication == RIGHT_POS_IND or match_any:
        if right_pos in the_dict['right_pos']:
            gotten_pos_val = the_dict['right_pos'][right_pos]['right_count']

            loc_vec = get_location_vec(RIGHT_POS_IND, words_list, loc)
            return gotten_pos_val, loc_vec

    if indication == NONE_POS_IND or match_any:
        gotten_pos_val = the_dict['wp_count']

        loc_vec = get_location_vec(NONE_POS_IND, words_list, loc)
        return gotten_pos_val, loc_vec

    return gotten_pos_val, loc_vec


def best_pos_in_loc(loc, the_dict, words_list, pos_list, indication, match_any, match_missing):
    best_pos = ''
    best_count = NONE_EXIST
    left_pos = None
    right_pos = None  

    if loc > 0:
        left_pos = pos_list[loc - 1]

    if loc + 1 < len(pos_list):
        right_pos = pos_list[loc + 1]
                    
    if match_missing:
        for _, word_values in the_dict.items():
            for pos, pos_values in word_values.items():
                val_tested, val_loc = match_value(pos_values, indication, right_pos, left_pos, match_any, words_list, loc)
                try:
                    if not isinstance(val_tested, int):
                       val_tested = cosine_similarity(val_tested.reshape(1,-1), val_loc.reshape(1,-1))[0][0]
                except:
                    import pdb; pdb.set_trace();
                    a=2

                if best_count < val_tested:
                    best_count = val_tested
                    best_pos = pos

    else:
        if words_list[loc] in the_dict.keys():
            for pos, pos_values in the_dict[words_list[loc]].items():
                val_tested, val_loc = match_value(pos_values, indication, right_pos, left_pos, match_any, words_list, loc)

                if not isinstance(val_tested, int):
                   val_tested = cosine_similarity(val_tested.reshape(1,-1), val_loc.reshape(1,-1))[0][0]

                if best_count < val_tested:
                    best_count = val_tested
                    best_pos = pos

    #         import pdb; pdb.set_trace();
    
    return best_count, best_pos


def get_pos_for_indication(all_possible_locs, indication, the_dict, words_list, pos_list, match_any, match_missing):
    specific_possible_locs = np.where(all_possible_locs == indication)[0]
    best_pos = ''
    best_count = NONE_EXIST
    best_loc = NONE_EXIST

    for loc in specific_possible_locs:
        tested_val, tested_pos = best_pos_in_loc(loc, the_dict, words_list, pos_list, indication, match_any, match_missing)
        if tested_pos != NONE_EXIST:
            if  best_count < tested_val:
                best_count = tested_val
                best_pos = tested_pos
                best_loc = loc
                
    return best_count, best_pos, best_loc


def get_best_pos(annotated_location, words_list, pos_list, the_dict, ma, ms):
    possible_locs = possible_location(annotated_location)

    best_count, best_pos, loc = get_pos_for_indication(possible_locs, BOTH_POS_IND, the_dict, words_list, pos_list, ma, ms)
    if best_count != NONE_EXIST:
        return best_pos, loc
    best_count, best_pos, loc = get_pos_for_indication(possible_locs, RIGHT_POS_IND, the_dict, words_list, pos_list, ma, ms)
    if best_count != NONE_EXIST:
        return best_pos, loc
    best_count, best_pos, loc = get_pos_for_indication(possible_locs, LEFT_POS_IND, the_dict, words_list, pos_list, ma, ms)
    if best_count != NONE_EXIST:
        return best_pos, loc
    best_count, best_pos, loc = get_pos_for_indication(possible_locs, NONE_POS_IND, the_dict, words_list, pos_list, ma, ms)
    return best_pos, loc

    
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-itr', '--input-train', type=str, default=POS_TRAIN_FILE, help='Input train file location')
    parser.add_argument('-id', '--input-dev', type=str, default=POS_DEV_FILE, help='Input dev set to calc accuracy')
    parser.add_argument('-it', '--input-test', type=str, default=POS_TEST_FILE, help='Input test set location')
    parser.add_argument('-ot', '--output-test', type=str, default=OUTPUT_FILE, help='Output Test set location')

    return parser.parse_args()


def make_pos_all_file(file_lines, words_pos_dict, dev_ind):
    lines_output = []
    matching_pos = 0
    total_pos = 0

    for _, file_line in enumerate(tqdm(file_lines)):
        match_any = False
        match_missing = False
        splitted_line = file_line.split() 
        len_line = len(splitted_line)
        annotated_location = np.zeros(len_line)
        annotated_token = [''] * len_line
        words_list, label_pos = split_to_lists(file_line, dev_ind)

        while not all(annotated_location):               
            best_pos, loc = get_best_pos(annotated_location, \
                                         words_list, \
                                         annotated_token, \
                                         words_pos_dict, \
                                         match_any, \
                                         match_missing)

            if match_any and (loc == NONE_EXIST):
                match_missing = True
                match_any = False

            elif loc == NONE_EXIST:
                match_any = True
            else:
                annotated_location[loc] = 1
                annotated_token[loc] = best_pos

        lines_output.append(' '.join([f'{i}/{j}' for i,j in zip(words_list,annotated_token)]))
        if dev_ind:
            matching_pos += sum(np.asarray(annotated_token)==np.asarray(label_pos))
            total_pos += len_line

    if dev_ind:
        acc = round(matching_pos/total_pos,3) * 100
        print(f'The accuracy over the data set is {acc}%')
        
    return lines_output


def main(args):  
    # Create the dictionary
    tic = time()
    words_pos_dict = make_dictonary(args.input_train)
    toc = time()
    print('Ceate dictionary running time: ', round((toc- tic)/60, 2))

    with open(os.path.join(BASE_LOC, r'dict_temp.pickle'), 'wb') as fp:
        pickle.dump(words_pos_dict, fp) 

    # Report accuracy on dev set
    tic = time()
    file_lines = read_file(args.input_dev)
    _ = make_pos_all_file(file_lines, words_pos_dict, 1)
    toc = time()
    print('Dev file running time: ', round((toc- tic)/60, 2))
    
    # Make redictions on train set
    tic = time()    
    file_lines = read_file(args.input_test)#POS_DEV_FILE)
    lines_output = make_pos_all_file(file_lines, words_pos_dict, 0)
    toc = time()
    print('Test file running time: ', round((toc- tic)/60, 2))
    
    # Save the results
    write_file(args.output_test, lines_output)
    

if __name__ == '__main__':
    arguments = parse_arguments()
    main(arguments)
