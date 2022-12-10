

VEC_MODEL_NAME = 'glove-twitter-25'
model_vectors = api.load(VEC_MODEL_NAME)

from collections import defaultdict

    
def read_file(location):
    file = open(location, "r")
    file_lines = file.readlines()
    file.close()
    return file_lines


def add_wp_to_dict(the_dict, word, pos, t2v):
    if word in the_dict:
        if pos in the_dict[word]:
            the_dict[word][pos]['wp_count'].append(t2v)
        else:
            the_dict[word][pos] = {'wp_count': [t2v], 'right_pos':{}, 'left_pos':{}}

    else:


        the_dict[word] = {pos: {'wp_count': [t2v], 'right_pos':{}, 'left_pos':{}}}


def add_right_pos_to_dict(the_dict, word, pos, right_pos, t2v):
    if right_pos in the_dict[word][pos]['right_pos']:
        the_dict[word][pos]['right_pos'][right_pos]['right_count'].append(t2v)
    else:
        the_dict[word][pos]['right_pos'][right_pos] = {'right_count':[t2v], 'left_pos':{}}
           

def add_left_pos_to_dict(the_dict, word, pos, left_pos, t2v):
    if left_pos in the_dict[word][pos]['left_pos']:
        the_dict[word][pos]['left_pos'][left_pos]['left_count'].append(t2v)
    else:
        the_dict[word][pos]['left_pos'][left_pos] = {'left_count': [t2v]}
        
        
def add_wp_right_pos_left_pos_to_dict(the_dict, word, pos, right_pos, left_pos, t2v):
    if left_pos in the_dict[word][pos]['right_pos'][right_pos]['left_pos']:
        the_dict[word][pos]['right_pos'][right_pos]['left_pos'][left_pos]['left_count'].append(t2v)
    else:
        the_dict[word][pos]['right_pos'][right_pos]['left_pos'][left_pos] = {'left_count': [t2v]} 

        
        
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


def convert_to_num(word):
    if word.isnumeric():
        return num2words(int(word))
    return word


def gec_vecs(token):
    t2search = convert_to_num(token.lower())

    if t2search in model_vectors.key_to_index.keys():
        t2v = model_vectors[t2search]

    else:
        chars_list = []
        
        for character in t2search:
            character2seach = convert_to_num(character)
            
            if character2seach in model_vectors.key_to_index.keys():
                chars_list.append(model_vectors[character2seach])

        t2v = np.mean(chars_list, axis=0)
        
    return t2v


def average_vecs(words_pos_dict):
    for word, word_values in words_pos_dict.items():
        for pos, pos_values in word_values.items():
            words_pos_dict[word][pos]['wp_count'] = np.mean(words_pos_dict[word][pos]['wp_count'], axis=0)
            
            if len(pos_values['right_pos']):
                for right_pos, right_poses_values in pos_values['right_pos'].items():
                    words_pos_dict[word][pos]['right_pos'][right_pos]['right_count'] = np.mean(right_poses_values['right_count'], axis=0)
                
                    if len(right_poses_values['left_pos']):
                        for rl_pos, rl_poses_values in right_poses_values['left_pos'].items():
                            words_pos_dict[word][pos]['right_pos'][right_pos]['left_pos'][rl_pos]['left_count'] = np.mean(rl_poses_values['left_count'], axis=0)
                
                
            if len(pos_values['right_pos']):
                for left_pos, left_poses_values in pos_values['left_pos'].items(): 
                    words_pos_dict[word][pos]['left_pos'][left_pos]['left_count'] = np.mean(left_poses_values['left_count'], axis=0)

    return words_pos_dict

           
def make_dictonary(train_location):
    file_lines = read_file(train_location)
    words_pos_dict = defaultdict()

    for file_line in tqdm(file_lines):
        splitted_line = file_line.split()
        len_line = len(splitted_line)
    
        for ii, _ in enumerate(splitted_line):
            token, pos = split_token_pos(splitted_line, ii)       
            t2v = gec_vecs(tokens)
            
            add_wp_to_dict(words_pos_dict, token, pos, t2v)
    
            if ii  + 1 < len_line:
                right_token, right_pos = split_token_pos(splitted_line, ii + 1)
                add_right_pos_to_dict(words_pos_dict, token, pos, right_pos, t2v)
    
            if ii - 1 <= 0:
                left_token, left_pos = split_token_pos(splitted_line, ii - 1)
                add_left_pos_to_dict(words_pos_dict, token, pos, left_pos, t2v)
    
            if (ii - 1 <= 0) and (ii  + 1 < len_line):
                add_wp_right_pos_left_pos_to_dict(words_pos_dict, token, pos, right_pos, left_pos, t2v)

    words_pos_dict = average_vecs(words_pos_dict):

    return words_pos_dict
