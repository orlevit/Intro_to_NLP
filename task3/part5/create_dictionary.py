from collections import defaultdict

    
def read_file(location):
    file = open(location, "r")
    file_lines = file.readlines()
    file.close()
    return file_lines


def add_wp_to_dict(the_dict, word, pos):
    if word in the_dict:
        if pos in the_dict[word]:
            the_dict[word][pos]['wp_count'] += 1
        else:
            the_dict[word][pos] = {'wp_count':1, 'right_pos':{}, 'left_pos':{}}

    else:
        the_dict[word] = {pos: {'wp_count':1, 'right_pos':{}, 'left_pos':{}}}


def add_right_pos_to_dict(the_dict, word, pos, right_pos):
    if right_pos in the_dict[word][pos]['right_pos']:
        the_dict[word][pos]['right_pos'][right_pos]['right_count'] += 1
    else:
        the_dict[word][pos]['right_pos'][right_pos] = {'right_count':1, 'left_pos':{}}
           

def add_left_pos_to_dict(the_dict, word, pos, left_pos):
    if left_pos in the_dict[word][pos]['left_pos']:
        the_dict[word][pos]['left_pos'][left_pos]['left_count'] += 1
    else:
        the_dict[word][pos]['left_pos'][left_pos] = {'left_count':1}
        
        
def add_wp_right_pos_left_pos_to_dict(the_dict, word, pos, right_pos, left_pos):
    if left_pos in the_dict[word][pos]['right_pos'][right_pos]['left_pos']:
        the_dict[word][pos]['right_pos'][right_pos]['left_pos'][left_pos]['left_count'] += 1
    else:
        the_dict[word][pos]['right_pos'][right_pos]['left_pos'][left_pos] = {'left_count':1} 

        
        
def split_token_pos(token_and_pos, loc):
    token, pos = token_and_pos[loc].rsplit('/',1)

    return token, pos

def make_dictonary(train_location):
    file_lines = read_file(train_location)
    words_pos_dict = defaultdict()
    
    for file_line in file_lines:
        splitted_line = file_line.split()
        len_line = len(splitted_line)
        
        for ii, _ in enumerate(splitted_line):
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
                
    return words_pos_dict