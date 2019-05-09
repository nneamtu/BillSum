# Methods to create final summaries froms cored sentences
import numpy as np 
import operator
from sklearn.metrics.pairwise import cosine_similarity

MAX_SUMMARY_LENGTH = 2000

def greedy_summarize(sent_texts, weights, threshold=15, return_idx=False,
                        summary_len = MAX_SUMMARY_LENGTH):

    # Sort indicies of weights
    sweights = np.argsort(weights)
    
    sent_lens = [len(s) for s in sent_texts]
   
    my_len = sum(sent_lens)

    #summary_len = int(my_len * .15)

 
    # See how many we can add until we reach limit 
    top_idx = [] 
    total_chars = 0

    for i in reversed(sweights):


        mylen = sent_lens[i] + 1 # Add 1 for space between sents

        # If sentence is too long - skip it
        if total_chars + mylen > summary_len:
            continue
        
        # if len(sent_texts[i].split()) < 8:
        #     continue # 2 short

        top_idx.append(i)
        total_chars += mylen

    # Put sentences into original document order
    final_idx = sorted(top_idx)
    
    if return_idx:
        return final_idx
        
    return np.array(sent_texts)[final_idx]



from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import minmax_scale

from functools import reduce
import operator
import re
word_counter = re.compile('[A-Za-z][A-Za-z]+')


def mmr_selection(sents, scores, doc_chars, max_fraction=0.15, L=0.7, min_words=5):
    '''
    Pick best sentences using MMR algo - up to 15% of words.
    "Query similarity" is the score.
    '''

    sent_wc = [reduce(operator.add, (1 for _ in word_counter.finditer(s)), 0) for s in sents]

    max_chars = int(doc_chars * max_fraction)

    # Rescale the scores and prepare sims 

    scores2 = minmax_scale(scores)
    cv = CountVectorizer(binary=True)
    sentX = cv.fit_transform(sents)
    sent_sims = cosine_similarity(sentX)
    final_sents = []
    used_sent_idx = []
    
    cur_chars = 0

    while cur_chars <= max_chars:
        cur_best = -1
        cur_max = -1
        cur_best_chars = 0
        
        for i in range(len(sents)):
            if i in used_sent_idx:
                continue

            if sent_wc[i] < min_words or '<SECTION-HEADER>' in sents[i]: #or 'This Act may' in sents[i]:
                continue

            mychars = len(sents[i])
            if cur_chars + mychars > max_chars:
                continue

            s1 = scores[i]
            # if s1 < 0.4:
                # continue
            # Find max sim of sentences already in sum
            if len(used_sent_idx) == 0:
                s2 = 0
            else:
                cands = sent_sims[i, used_sent_idx]
                s2 = cands.max()
            
            my_score = L * s1 - (1-L) * s2
            
            if my_score > cur_max:
                cur_max = my_score
                cur_best = i
                cur_best_chars = mychars
                
        if cur_best == -1:
            break # No sentences left to add 
        
        used_sent_idx.append(cur_best)
        cur_chars += cur_best_chars
        final_sents.append(sents[cur_best])

    return final_sents #, used_sent_idx
