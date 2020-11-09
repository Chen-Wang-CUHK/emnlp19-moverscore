from __future__ import division
import os
import json
import codecs
import copy
import torch
import re

import sys
sys.path.append('../')

import mvrs_config

# get the general configuration
parser = mvrs_config.ArgumentParser("run_summarization.py")
mvrs_config.general_args(parser)
opt = parser.parse_args()

if opt.mvrs_type == 'v1':
    from moverscore import get_idf_dict, word_mover_score
else:
    assert opt.mvrs_type == 'v2'
    from moverscore_v2 import get_idf_dict, word_mover_score, plot_example, MAX_POSITION

BASE_FOLDER = "data"
name = opt.dataset
year = name.split('.')[1]
assert year in ['08', '09', '2010', '2011', 'cnndm']


def load_json(filename):
    filepath = os.path.join(BASE_FOLDER, filename)
    with codecs.open(filepath, 'r', encoding='utf-8') as f:
        return json.loads(f.read())


def normalize_responsiveness(dataset):
    max_resp = 0.
    for k,v in dataset.items():
        for annot in v['annotations']:
            if annot['responsiveness'] > max_resp:
                max_resp = annot['responsiveness']
    for k,v in dataset.items():
        for annot in v['annotations']:
            annot['responsiveness'] /= float(max_resp)
    return dataset


if year != 'cnndm':
    dataset_mds_gen_resp_pyr = normalize_responsiveness(load_json(name))
else:
    dataset_mds_gen_resp_pyr = load_json(name)

# # # For debug, compare the generated tac_09 and the original tac_09
# # /****
#
# gen_tac_09_mds_gen_resp_pyr = normalize_responsiveness(load_json("tac.09.trueRef.withTrueRefsInSys.mds.gen.resp-pyr"))
#
# def preprocess_one_line(line):
#     line = line.strip()
#     orig_words = line.split()
#     new_words = []
#     for w in orig_words:
#         # 1. remove "___"
#         if len(w) >= 2 and all([c == '_' for c in w]):
#             continue
#         # 2. "insurgent.Four" --> " insurgent. Four"
#         posis = [p.start() for p in re.finditer("\.",w)]
#         if len(posis) == 1 and w[-1] != '.' and w[posis[0]+1].isupper() and len(w) > 3:
#             w = w.replace('.', '. ')
#         new_words.append(w)
#     new_line = ' '.join(new_words)
#     return new_line
#
# for topic_key in gen_tac_09_mds_gen_resp_pyr:
#     # compare references
#     gen_refs = gen_tac_09_mds_gen_resp_pyr[topic_key]['references']
#     orig_refs = tac_09_mds_gen_resp_pyr[topic_key]['references']
#     gen_refs_str = [' '.join(sents['text']) for sents in gen_refs]
#     orig_refs_str = [' '.join(sents['text']) for sents in orig_refs]
#     for one_gen_ref_str in gen_refs_str:
#         assert one_gen_ref_str in orig_refs_str, '\n{}: the original_refs_str:\n{}\ndoes not contain the one_gen_ref_str:\n{}\n'.format(topic_key, orig_refs_str, one_gen_ref_str)
#     for one_orig_ref_str in orig_refs_str:
#         assert one_orig_ref_str in gen_refs_str, '\n{}: the generated_refs_str:\n{}\ndoes not contain the one_orig_ref_str:\n{}\n'.format(topic_key, gen_refs_str, one_orig_ref_str)
#
#     # compare annotations
#     gen_annots = gen_tac_09_mds_gen_resp_pyr[topic_key]['annotations']
#     orig_annots = tac_09_mds_gen_resp_pyr[topic_key]['annotations']
#     assert (len(gen_annots) == len(orig_annots) - len(orig_refs)) or (len(gen_annots) == len(orig_annots))
#     for i in range(len(gen_annots)):
#         one_gen_annots = gen_annots[i]
#         one_orig_annots = orig_annots[i]
#         # err_msg = '{}_annot_sum{}:\none_gen_annots:\n{}\none_orig_annots:\n{}'.format(topic_key, i+1, one_gen_annots, one_orig_annots)
#         err_msg = '{}_annot_sum{}:\none_gen_annots:\n{}\none_orig_annots:\n{}'.format(topic_key, i+1, one_gen_annots['text'], one_orig_annots['text'])
#         err_msg = err_msg + '\none_gen_annots:\n{}\none_orig_annots:\n{}'.format(' '.join(one_gen_annots['text']), ' '.join(one_orig_annots['text']))
#         for annot_key in one_gen_annots:
#             if annot_key == 'text':
#                 one_gen_annots_text = ' '.join(one_gen_annots[annot_key]).strip().replace(',', '').replace('.','').split()
#                 one_gen_annots_set = set(one_gen_annots_text)
#                 one_orig_annots_text = preprocess_one_line(' '.join(one_orig_annots[annot_key])).strip().replace(',', '').replace('.','').split()
#                 one_orig_annots_set = set(one_orig_annots_text)
#                 if len(one_gen_annots_set) == 0:
#                     assert len(one_orig_annots_set) == 0, 'Inconsistent key: {}'.format(annot_key) + err_msg
#                 else:
#                     jc_sim = len(one_gen_annots_set & one_orig_annots_set) * 1.0 / len(one_gen_annots_set | one_orig_annots_set)
#                     if jc_sim != 1:
#                         print(topic_key + '\n' + err_msg + '\n')
#                     assert jc_sim >= 0.93, 'Inconsistent key: {}'.format(annot_key) + err_msg
#             elif annot_key == 'summ_id':
#                 assert str(one_gen_annots[annot_key]) == one_orig_annots[annot_key], 'Inconsistent key: {}'.format(
#                     annot_key) + err_msg
#             else:
#                 assert one_gen_annots[annot_key] == one_orig_annots[annot_key], 'Inconsistent key: {}'.format(annot_key) + err_msg
# print('here')
# # ****/


def merge_datasets(lst_datasets):
    merged_dataset = {}
    for dataset in lst_datasets:
        merged_dataset.update(copy.deepcopy(dataset))
    return merged_dataset

import pprint
pp = pprint.PrettyPrinter(indent=4)
import numpy as np
def print_average_correlation(corr_mat, human_metric, ave_type='macro'):
    corr_mat = np.array(corr_mat)   
    results = dict(zip([human_metric + '_' + ave_type + '_kendall', human_metric + '_' + ave_type + '_pearson', human_metric + '_' + ave_type + '_spearman'],
                       [np.mean(corr_mat[:,0]), 
                       np.mean(corr_mat[:,1]),
                       np.mean(corr_mat[:,2])]))
    pp.pprint(results)

if year != 'cnndm':
    tac_resp_data = merge_datasets([dataset_mds_gen_resp_pyr])
    tac_pyr_data = merge_datasets([dataset_mds_gen_resp_pyr])

    tac_pyr_data = dict(list(tac_pyr_data.items()))
    tac_resp_data = dict(list(tac_resp_data.items()))

    human_scores = ['pyr_score', 'responsiveness']
    dataset = [list(tac_pyr_data.items()), list(tac_resp_data.items())]
else:
    cnndm_overall_data = merge_datasets([dataset_mds_gen_resp_pyr])
    cnndm_grammar_data = merge_datasets([dataset_mds_gen_resp_pyr])
    cnndm_redundancy_data = merge_datasets([dataset_mds_gen_resp_pyr])
    human_scores = ['overall', 'grammar', 'redundancy']
    dataset = [list(cnndm_overall_data.items()), list(cnndm_grammar_data.items()), list(cnndm_redundancy_data.items())]

with open('stopwords.txt', 'r', encoding='utf-8') as f:
    stop_words = set(f.read().strip().split(' '))

import scipy.stats as stats
from tqdm import tqdm 

def micro_macro_averaging(dataset, target, device='cuda:0'):
    references, summaries = [], []
    for topic in dataset:
        k,v = topic
        references.extend([' '.join(ref['text']) for ref in v['references']])
        summaries.extend([' '.join(annot['text']) for annot in v['annotations']])
 
    idf_dict_ref = get_idf_dict(references)
    idf_dict_hyp = get_idf_dict(summaries)

    correlations = []
    annot_sent_cnt = {0: 0, 1: 0}
    total_hss = []
    total_pss = []
    for topic in tqdm(dataset):
        k,v = topic
        references = [' '.join(ref['text']) for ref in v['references']]
        num_refs = len(references)
        target_scores, prediction_scores = [], []      

        for annot in v['annotations']:
            # changed by wchen: '>' -> '>=' to include the one sentence annotations
            if len(annot['text']) >= 1:
                if len(annot['text']) == 1:
                    annot_sent_cnt[1] += 1
                target_scores.append(float(annot[target]))

                scores = word_mover_score(references, [' '.join(annot['text'])] * num_refs, idf_dict_ref, idf_dict_hyp, stop_words,
                                          n_gram=1, remove_subwords=True, batch_size=48, device=device)

                prediction_scores.append(np.mean(scores))
            else:
                annot_sent_cnt[0] += 1
        total_hss.extend(target_scores)
        total_pss.extend(prediction_scores)
        # skip the case that has equal human scores for each system
        if not (np.array(target_scores) == target_scores[0]).all():
            correlations.append([
                             stats.kendalltau(target_scores, prediction_scores)[0],
                             stats.pearsonr(target_scores, prediction_scores)[0],
                             stats.spearmanr(target_scores, prediction_scores)[0]])
    # macro averaged correlation scores
    print('\n annot_sent_cnt: {} \n'.format(annot_sent_cnt))
    print('\n=====ALL Macro RESULTS=====')
    print_average_correlation(correlations, human_metric=target, ave_type='macro')
    # micro averaged correlation scores
    micro_corr = [stats.kendalltau(total_hss, total_pss)[0],
                  stats.pearsonr(total_hss, total_pss)[0],
                  stats.spearmanr(total_hss, total_pss)[0]]
    print('\n=====ALL Micro RESULTS=====')
    print_average_correlation([micro_corr], human_metric=target, ave_type='micro')
    # return np.array(correlations)


if __name__ == '__main__':
    print("\nScript: run_summarization.py")
    print("Configurations:", opt)
    for i in range(len(human_scores)):
        print(human_scores[i])
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        bert_corr = micro_macro_averaging(dataset[i], human_scores[i], device=device)
