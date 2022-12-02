import os
import pickle
import argparse
import numpy as np
import pandas as pd
from time import time
from collections import defaultdict

CHUNCK = 2048
CONVERT_WIKI_IND = True
BASE_LOC = r'/RG/rg-tal/orlev/study/bar_ilan' 
WIKI_ORG_FILE_LOC = os.path.join(BASE_LOC, 'wikipedia.deps')
CONVERTED_WIKI_FILE_LOC = os.path.join(BASE_LOC, 'wiki_converted.pkl')
#WIKI_WORDFORM_FILE_LOC = os.path.join(BASE_LOC, 'wordform_small.pkl')
#WIKI_LEMMA_FILE_LOC = os.path.join(BASE_LOC, 'lemma_small.pkl')
#WIKI_POS_FILE_LOC = os.path.join(BASE_LOC, 'pos_small.pkl')
COUNTS_WORDS_FILE = os.path.join(BASE_LOC, 'counts_words.txt')
COUNTS_CONTEXTS_POS_FILE = os.path.join(BASE_LOC, 'counts_contexts_pos.txt')
TOP_SIMILARITIES_LOC = os.path.join(BASE_LOC, 'top20.csv')
TOP_FEATURES_LOC = os.path.join(BASE_LOC, 'top20_features.csv')

CC_MATRIX1 = {'tw_func_ind':True, 'cw_func_ind':False, 'pos_ind':False, 'window':np.inf, 'word_filter':100, 'feature_filter':75}
CC_MATRIX2 = {'tw_func_ind':False, 'cw_func_ind':False, 'pos_ind':False, 'window':2, 'word_filter':100, 'feature_filter':75}
CC_MATRIX3 = {'tw_func_ind':True, 'cw_func_ind':True, 'pos_ind':True, 'window':2, 'word_filter':100, 'feature_filter':75}

NON_FUNCTINAL = ['NN','JJ','NCD','NNP','NNT','RB','VB','CD','CDT','JJT','VB-TOINFINITIVE']
WORDS_TO_EVALUATE = ['מכונית', 'אוטובוס', 'משקה', 'מלון','רובה','פצצה','סוס','שועל','שולח','קערה','גיטרה', 'פסנתר']

def sanity_checks(ccw, wc, fc):
    # sanity check1
    total_wc = total_pairs(wc)
    total_fc = total_pairs(fc)
    if total_fc != total_wc :
       raise Exceptio("Feature dict total is not like words dict total!")

    # sanity check2
    total_cc = sum([freq for _, feas in ccw.items()  for _, freq in feas.items()])
    if total_cc != total_wc :
       raise Exceptio("cc matrix dict total is not like words dict total!")

    return total_cc

def total_pairs(freq_dict):
    total = 0
    for k, freq in freq_dict.items():
        total += freq

    return total

def calc_pmi(ccw, wc, fc):
    # sanity checks
    total_freq = sanity_checks(ccw, wc, fc)
    pmi_matrix = defaultdict()
    
    for word_num, features in ccw.items():
        pmi_matrix[word_num] = {}
        freq_word = wc[word_num]
        
        for feature_num, freq in features.items():
            freq_feature = fc[feature_num]
            pmi_matrix[word_num][feature_num] = np.log2((freq / (freq_word * freq_feature)) * total_freq)

            
    return pmi_matrix 

def top_similarities(pmi_matrix, ccf, selected_word_text, words_to_number, top_sim=20):
    def clac_numerator(pmi_matrix, ccf, filtered_number_mapping, selected_word):
        numerator = np.zeros(len(pmi_matrix))
        req_word = pmi_matrix[selected_word]
        
        for word_feature, pmi1 in req_word.items():
            for other_word, _ in ccf[word_feature].items():
                pmi2 = pmi_matrix[other_word][word_feature]
                numerator[filtered_number_mapping[other_word]] += pmi1 * pmi2

        return numerator

    def clac_denominator(pmi_matrix, filtered_number_mapping):
        denominator = np.zeros(len(pmi_matrix))
            
        for word_feature_num, features in pmi_matrix.items():
            for feature, pmi in features.items():
                denominator[filtered_number_mapping[word_feature_num]] += pmi ** 2
                
        denominator = np.sqrt(denominator)
            
        return denominator
    
    filtered_number_mapping = {num : i for i, num in enumerate(pmi_matrix)}
    filtered_number_mapping_fliped = {i :num for i, num in enumerate(pmi_matrix)}

    selected_word = words_to_number[selected_word_text]
    denominator = clac_denominator(pmi_matrix, filtered_number_mapping)
    numerator = clac_numerator(pmi_matrix, ccf, filtered_number_mapping, selected_word)
            
    denominator_selected_word = denominator[filtered_number_mapping[selected_word]]
    for i, _ in enumerate(denominator):
        denominator[i] *= denominator_selected_word

    cos_similarity = numerator / denominator
    cos_similarity[filtered_number_mapping[selected_word]] = -1
    most_similar = np.argsort(cos_similarity)[-top_sim:][::-1]
    most_similar_in_org_index = [filtered_number_mapping_fliped[i] for i in  most_similar]
    return most_similar_in_org_index

def saving(file_loc, dest_loc):
    file = open(file_loc, "r")
    sentence = []
    sentences = []
    file_chunk = file.readlines(CHUNCK)
    
    tic = time()
    while file_chunk:
        for line in file_chunk:
            if line == '\n':
                sentences.append(sentence)
                sentence = []
            else:
                splitted_line = line.split()
                sentence.append((splitted_line[2], splitted_line[3]))
        file_chunk = file.readlines(CHUNCK)
        
    if len(sentence):
        sentences.append(sentence)
    file.close()
    toc = time()
    sentence_len = len(sentences)
    
    print('File reading time(minutes): ', round((toc- tic)/60, 2),'Sentences: ', sentence_len)
    
    tic = time()
    with open(dest_loc, "wb" ) as f:
        pickle.dump(sentences, f)
    toc = time()
    print('Saving time(minutes): ', round((toc- tic)/60, 2))

def get_word_with_position(word, pos_ind, position):
    if pos_ind:
        return word + ':' + str(position)
    return word

def add_count_to_dict(the_dict, word_num):
    if word_num in the_dict:
        the_dict[word_num] += 1
    else:
        the_dict[word_num] = 1
    
def get_to_word_number(word, words_to_number):
    if word not in words_to_number:
        words_to_number[word] = len(words_to_number)
    return words_to_number[word]
        
def add_to_cc_matrix(co_occurence_matrix, tw_num, cw_num):
    if tw_num not in co_occurence_matrix:
        co_occurence_matrix[tw_num] = {}                

    add_count_to_dict(co_occurence_matrix[tw_num], cw_num)   
        
    
def add_to_matrix(tw_num, sentence, context_indices, co_occurence_matrix, words_to_number, tw_i, features_to_words, words_count, features_count, pos_ind):
    for cw_i in context_indices: 
        cw = get_word_with_position(sentence[cw_i][0], pos_ind, cw_i-tw_i)
        cw_num = get_to_word_number(cw, words_to_number)
        add_to_cc_matrix(co_occurence_matrix, tw_num, cw_num)
        add_to_cc_matrix(features_to_words, cw_num, tw_num)
        add_count_to_dict(features_count, cw_num)
        
def rel_sentence_indices(sentence, functional_ind):
    if functional_ind:
        return list(range(len(sentence)))
    
    return [ w_i for w_i, word in enumerate(sentence) if word[-1] in NON_FUNCTINAL ]
   
def rel_window_indices(tw_i, cw_sentence_indices, context_window):
    cw_sentence_indices = np.asarray(cw_sentence_indices)
    left_indices = cw_sentence_indices[cw_sentence_indices<tw_i]
    right_indices = cw_sentence_indices[cw_sentence_indices>tw_i]
   
    left_limit = 0 if context_window == np.inf else - context_window
    left_indices_window = left_indices[left_limit:]
    right_indices_window = right_indices[:min(context_window, len(cw_sentence_indices))]
    context_indices = list(left_indices_window) + list(right_indices_window)

    return context_indices 

def calc_co_occurence_matrix(sentences, context_window, tw_func_ind=False, cw_func_ind=False, pos_ind=None):
    words_count = defaultdict() 
    features_count = defaultdict() 
    words_to_number = defaultdict()
    features_to_words = defaultdict()
    co_occurence_matrix = defaultdict()
    
    for sentence in sentences:
        tw_sentence_indices = rel_sentence_indices(sentence, tw_func_ind)
        cw_sentence_indices = rel_sentence_indices(sentence, cw_func_ind)

        for tw_i in tw_sentence_indices:
            tw = get_word_with_position(sentence[tw_i][0], pos_ind, 0)
            tw_num = get_to_word_number(tw, words_to_number)
            add_count_to_dict(words_count, tw_num)
            context_indices = rel_window_indices(tw_i, cw_sentence_indices, context_window)
            add_to_matrix(tw_num, sentence, context_indices, co_occurence_matrix, words_to_number, tw_i, features_to_words, words_count, features_count, pos_ind)
        
    return co_occurence_matrix, words_count, features_to_words, features_count, words_to_number


def threshold_filtering(wc, fc, word_occurances_limit, feature_occurances_limit):
    filtered_wc = {k:v for k,v in wc.items() if word_occurances_limit <= v}
    filtered_fc = {k:v for k,v in fc.items() if feature_occurances_limit <= v}
    
    return filtered_wc, filtered_fc

def threshold_matrix(matrix, fwc_tmp, ffc_tmp):
    fwc = {}
    temp_fwc = {}
    temp_matrix = {}
    filtred_matrix = {}
    
    for word, w_occurances in fwc_tmp.items(): 
        temp_fwc[word] = 0 
        temp_matrix[word] = {}

        for feature, f_occurances in matrix[word].items(): 
            if feature in ffc_tmp:
                temp_fwc[word] += f_occurances
                temp_matrix[word][feature] = f_occurances
                    
    for word, freq in temp_fwc.items(): 
        if 0 < freq:
            fwc[word] = freq

    for word, features in temp_matrix.items(): 
        if 0 < len(features):
            filtred_matrix[word] = features
            
    return filtred_matrix, fwc

def save_counts(wc1, words_to_number, loc, top_common):
    number_to_number = {v: k for k, v in words_to_number.items()}

    most_freq = list(sorted(wc1.items(), key=lambda item: item[1], reverse=True))[:top_common]
    lines = [number_to_number[line[0]] + ' ' + str(line[1]) + '\n' for line in most_freq]
        
    with open(loc, "w") as f:
        f.writelines(lines)
        
def filtered_cc(sentences, tw_func_ind, cw_func_ind, pos_ind, window, word_filter=100, feature_filter=75):
    cc_w_matrix, wc, cc_f_matrix, fc, words_to_number = calc_co_occurence_matrix(sentences=sentences,
                                                                                 context_window=window,
                                                                                 tw_func_ind=tw_func_ind,
                                                                                 cw_func_ind=cw_func_ind, 
                                                                                 pos_ind=pos_ind)
    
    fwc_tmp, ffc_tmp = threshold_filtering(wc, fc, word_filter, feature_filter)
    cc_filtered_w_matrix, fwc = threshold_matrix(cc_w_matrix, fwc_tmp, ffc_tmp)
    cc_filtered_f_matrix, ffc = threshold_matrix(cc_f_matrix, ffc_tmp, fwc_tmp)
    return cc_filtered_w_matrix, fwc, cc_filtered_f_matrix, ffc, words_to_number

def top_words_similarity(pmi_matrix, ccf, selected_word_text, words_to_number, top_sim):
    similarity = top_similarities(pmi_matrix, ccf, selected_word_text, words_to_number, top_sim)
    number_to_words = {v:k for k,v in words_to_number.items()}
    return [ number_to_words[word_num] for word_num in similarity ]

def add_lists_to_df(list1, list2, list3, selected_word_text, df):

        df_section = pd.DataFrame({})
        df_section['co-occurrence1'] = list1
        df_section['co-occurrence2'] = list2
        df_section['co-occurrence3'] = list3
    
        df.loc[len(df)] = [selected_word_text, '', '']
        df.loc[len(df)] = ['***********'] * 3
        df = df.append(df_section)
        df.loc[len(df)] = [''] * 3

        return df

def save_top_similar_words(pmi_matrix1, ccf1, words_to_number1, \
                           pmi_matrix2, ccf2, words_to_number2, \
                           pmi_matrix3, ccf3, words_to_number3,
                           words_to_evaluate, similarities_loc, top_sim):

    df = pd.DataFrame({}, columns=['co-occurrence1', 'co-occurrence2','co-occurrence3'])
    for selected_word_text in words_to_evaluate:
        df_section = pd.DataFrame({})
    
        words1 = top_words_similarity(pmi_matrix1, ccf1, selected_word_text, words_to_number1, top_sim)
        words2 = top_words_similarity(pmi_matrix2, ccf2, selected_word_text, words_to_number2, top_sim)
        words3 = top_words_similarity(pmi_matrix3, ccf3, selected_word_text+':0', words_to_number3, top_sim)

        df = add_lists_to_df(words1, words2, words3, selected_word_text, df)

    df.to_csv(similarities_loc)

def get_top_features(pmi_matrix, selected_word_text, words_to_number, top_features):
    selected_word = words_to_number[selected_word_text]
    top_features_list = sorted(pmi_matrix[selected_word].items(), key=lambda x: x[-1], reverse=True)[:top_features]
    number_to_words = {v:k for k,v in words_to_number.items()}

    return [ number_to_words[feature_pair[0]] for feature_pair in top_features_list ]

def save_top_features(pmi_matrix1, pmi_matrix2, pmi_matrix3,  words_to_number1, words_to_number2, words_to_number3, \
                      words_to_evaluate, top_features_loc, top_features=20):

    df = pd.DataFrame({}, columns=['co-occurrence1', 'co-occurrence2','co-occurrence3'])
    for selected_word_text in words_to_evaluate:
        df_section = pd.DataFrame({})
    
        features1 = get_top_features(pmi_matrix1, selected_word_text, words_to_number1, top_features)
        features2 = get_top_features(pmi_matrix2, selected_word_text, words_to_number2, top_features)
        features3 = get_top_features(pmi_matrix3, selected_word_text+':0', words_to_number3, top_features)

        df = add_lists_to_df(features1, features2, features3, selected_word_text, df)

    df.to_csv(top_features_loc)

def words_features_statistics(pmi_matrix, fc, text):
    words_count = 0
    features_list = []

    for word, features in pmi_matrix.items():
        words_count += 1
        features_list.append(len(features))     

    toatl_features = len(fc)
    max_features = max(features_list)
    min_features = min(features_list)
    mean_features = np.mean(features_list)
    print('----------------- ' + text + ' statistics -------------------')
    #print(text + ' statistics')
    print('Words: ', words_count, '\nTotal features: ', toatl_features, '\nMax features: ', max_features,
          '\nMin features: ', min_features, '\nMean features: ', mean_features)
    print('--------------------------------------------------------------')

def load_file(src_loc, target_loc, convert_wiki_ind=False):
    if convert_wiki_ind:
       saving(src_loc, target_loc)

    with open(target_loc, "rb" ) as f:
        sentences = pickle.load(f)

    return sentences

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 'True', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'False', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-io', '--input-original', type=str, default=WIKI_ORG_FILE_LOC, help='The originl wiki file')
    parser.add_argument('-ic', '--input-converted', type=str, default=CONVERTED_WIKI_FILE_LOC, help='Converted wiki file location(only required columns out of the file)')
    parser.add_argument('-ci', '--convert-indication', type=str2bool, default=True, help='Whether the converted file already exist. No conversonis needed')
    parser.add_argument('-tw', '--top-words-file', default=TOP_SIMILARITIES_LOC, type=str, help='The location of the top wordsfile')
    parser.add_argument('-tf', '--top-features-file', default=TOP_FEATURES_LOC, type=str, help='The location of the top features file')
    parser.add_argument('-cw', '--common-words-file', default=COUNTS_WORDS_FILE, type=str, help='The location of the common word file')
    parser.add_argument('-cf', '--common-features-file', default=COUNTS_CONTEXTS_POS_FILE, type=str, help='The location of the common features file')
    parser.add_argument('-tv', '--top-values', default=20, type=int, help='Top X values in features/similarities')
    parser.add_argument('-tc', '--top-counts', default=50, type=int, help='Top X common features/similarities')

    return parser.parse_args()

def main(args):
    tic = time()
    sentences = load_file(args.input_original, args.input_converted, args.convert_indication)
    
    ccw1, wc1, ccf1, fc1, words_to_number1 = filtered_cc(sentences, **CC_MATRIX1)
    ccw2, wc2, ccf2, fc2, words_to_number2 = filtered_cc(sentences, **CC_MATRIX2)
    ccw3, wc3, ccf3, fc3, words_to_number3 = filtered_cc(sentences, **CC_MATRIX3)
    
    save_counts(wc1, words_to_number1, args.common_words_file, args.top_counts)
    save_counts(fc3, words_to_number3, args.common_features_file, args.top_counts)
    
    
    pmi_matrix1 = calc_pmi(ccw1, wc1, fc1)
    pmi_matrix2 = calc_pmi(ccw2, wc2, fc2)
    pmi_matrix3 = calc_pmi(ccw3, wc3, fc3)
    
    words_features_statistics(pmi_matrix1, fc1, 'co-occurrence1')
    words_features_statistics(pmi_matrix2, fc2, 'co-occurrence2')
    words_features_statistics(pmi_matrix3, fc3, 'co-occurrence3') 

    save_top_similar_words(pmi_matrix1, ccf1, words_to_number1, \
                           pmi_matrix2, ccf2, words_to_number2, \
                           pmi_matrix3, ccf3, words_to_number3,
                           WORDS_TO_EVALUATE, args.top_words_file, args.top_values)
    
    save_top_features(pmi_matrix1, pmi_matrix2, pmi_matrix3, words_to_number1, words_to_number2, words_to_number3, WORDS_TO_EVALUATE, 
                      args.top_features_file, args.top_values)

    toc = time()
    print('Total running time: ', round((toc- tic)/60, 2))

if __name__ == '__main__':
    args = parse_arguments()
    main(args)
