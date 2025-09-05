"""
For 4.3. we need sublexica.

Written with the help of GenAI
"""
import random

def get_sub_lexica(number_of_words, lexicon_name):
    if lexicon_name == 'Warriner':
        range_int = 8422
        path = "Warriner_refined_withoutGOLD_limited_to_COHA"
    if lexicon_name == 'NRC_VAD':
        range_int = 10906
        path = "NRC_VAD_refined_withoutGOLD_limited_to_COHA"

    base_lexicon = []
    with open(path) as f:
        for l in f:
            l = l.split('\t')
            base_lexicon.append([l[0], float(l[1]), float(l[2]), float(l[3])])

    random.seed(42)
    for i in range(50):
        list1 = random.sample(range(0, range_int), k=number_of_words)
        result_lexicon = [base_lexicon[i] for i in list1]

        with open(lexicon_name+str(number_of_words)+'/'+lexicon_name+'_'+str(number_of_words)+'_'+str(i+1),mode='w') as f:
            for row in result_lexicon:
                l = '\t'.join(map(str, row))
                f.write(l + '\n')

for i in [2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]:
    get_sub_lexica(i, 'NRC_VAD')



