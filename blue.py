import json
import os
import argparse

from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
'''
This script was adapted from the original version by hieuhoang1972 which is part of MOSES. 
'''

'''Provides:

cook_refs(refs, n=4): Transform a list of reference sentences as strings into a form usable by cook_test().
cook_test(test, refs, n=4): Transform a test sentence as a string (together with the cooked reference sentences) into a form usable by score_cooked().
score_cooked(alltest, n=4): Score a list of cooked test sentences.

score_set(s, testid, refids, n=4): Interface with dataset.py; calculate BLEU score of testid against refids.

The reason for breaking the BLEU computation into three phases cook_refs(), cook_test(), and score_cooked() is to allow the caller to calculate BLEU scores for multiple test sets as efficiently as possible.
'''

import sys, math, re, xml.sax.saxutils
import subprocess
import os
import nltk

nonorm = 0

preserve_case = False
eff_ref_len = "shortest"

normalize1 = [
    ('<skipped>', ''),  # strip "skipped" tags
    (r'-\n', ''),  # strip end-of-line hyphenation and join lines
    (r'\n', ' '),  # join lines
    #    (r'(\d)\s+(?=\d)', r'\1'), # join digits
]
normalize1 = [(re.compile(pattern), replace) for (pattern, replace) in normalize1]

normalize2 = [
    (r'([\{-\~\[-\` -\&\(-\+\:-\@\/])', r' \1 '),  # tokenize punctuation. apostrophe is missing
    (r'([^0-9])([\.,])', r'\1 \2 '),  # tokenize period and comma unless preceded by a digit
    (r'([\.,])([^0-9])', r' \1 \2'),  # tokenize period and comma unless followed by a digit
    (r'([0-9])(-)', r'\1 \2 ')  # tokenize dash when preceded by a digit
]
normalize2 = [(re.compile(pattern), replace) for (pattern, replace) in normalize2]


def normalize(s):
    '''Normalize and tokenize text. This is lifted from NIST mteval-v11a.pl.'''
    # Added to bypass NIST-style pre-processing of hyp and ref files -- wade
    if (nonorm):
        return s.split()
    if type(s) is not str:
        s = " ".join(s)
    # language-independent part:
    for (pattern, replace) in normalize1:
        s = re.sub(pattern, replace, s)
    s = xml.sax.saxutils.unescape(s, {'&quot;': '"'})
    # language-dependent part (assuming Western languages):
    s = " %s " % s
    if not preserve_case:
        s = s.lower()  # this might not be identical to the original
    for (pattern, replace) in normalize2:
        s = re.sub(pattern, replace, s)
    return s.split()


def count_ngrams(words, n=4):
    counts = {}
    for k in range(1, n + 1):
        for i in range(len(words) - k + 1):
            ngram = tuple(words[i:i + k])
            counts[ngram] = counts.get(ngram, 0) + 1
    return counts


def cook_refs(refs, n=4):
    '''Takes a list of reference sentences for a single segment
    and returns an object that encapsulates everything that BLEU
    needs to know about them.'''

    refs = [normalize(ref) for ref in refs]
    maxcounts = {}
    for ref in refs:
        counts = count_ngrams(ref, n)
        for (ngram, count) in counts.items():
            maxcounts[ngram] = max(maxcounts.get(ngram, 0), count)
    return ([len(ref) for ref in refs], maxcounts)


def cook_test(test, item, n=4):
    '''Takes a test sentence and returns an object that
    encapsulates everything that BLEU needs to know about it.'''
    (reflens, refmaxcounts) = item
    test = normalize(test)
    result = {}
    result["testlen"] = len(test)

    # Calculate effective reference sentence length.

    if eff_ref_len == "shortest":
        result["reflen"] = min(reflens)
    elif eff_ref_len == "average":
        result["reflen"] = float(sum(reflens)) / len(reflens)
    elif eff_ref_len == "closest":
        min_diff = None
        for reflen in reflens:
            if min_diff is None or abs(reflen - len(test)) < min_diff:
                min_diff = abs(reflen - len(test))
                result['reflen'] = reflen

    result["guess"] = [max(len(test) - k + 1, 0) for k in range(1, n + 1)]

    result['correct'] = [0] * n
    counts = count_ngrams(test, n)
    for (ngram, count) in counts.items():
        result["correct"][len(ngram) - 1] += min(refmaxcounts.get(ngram, 0), count)

    return result


def score_cooked(allcomps, n=4, ground=0, smooth=1):
    totalcomps = {'testlen': 0, 'reflen': 0, 'guess': [0] * n, 'correct': [0] * n}
    for comps in allcomps:
        for key in ['testlen', 'reflen']:
            totalcomps[key] += comps[key]
        for key in ['guess', 'correct']:
            for k in range(n):
                totalcomps[key][k] += comps[key][k]
    logbleu = 0.0
    all_bleus = []
    for k in range(n):
        correct = totalcomps['correct'][k]
        guess = totalcomps['guess'][k]
        addsmooth = 0
        if smooth == 1 and k > 0:
            addsmooth = 1
        logbleu += math.log(correct + addsmooth + sys.float_info.min) - math.log(guess + addsmooth + sys.float_info.min)
        if guess == 0:
            all_bleus.append(-10000000)
        else:
            all_bleus.append(math.log(correct + sys.float_info.min) - math.log(guess))

    logbleu /= float(n)
    all_bleus.insert(0, logbleu)

    brevPenalty = min(0, 1 - float(totalcomps['reflen'] + 1) / (totalcomps['testlen'] + 1))
    for i in range(len(all_bleus)):
        if i == 0:
            all_bleus[i] += brevPenalty
        all_bleus[i] = math.exp(all_bleus[i])
    return all_bleus


def bleu(refs, candidate, ground=0, smooth=1):
    refs = cook_refs(refs)
    test = cook_test(candidate, refs)
    return score_cooked([test], ground=ground, smooth=smooth)


def splitPuncts(line):
    return ' '.join(re.findall(r"[\w]+|[^\s\w]", line))


def bleu_fromstr(predictions, golds, rmstop=True):
    predictions = [" ".join(nltk.wordpunct_tokenize(predictions[i])) for i in range(len(predictions))]
    golds = [" ".join(nltk.wordpunct_tokenize(g)) for g in golds]
    if rmstop:
        pypath = os.path.dirname(os.path.realpath(__file__))
        stopwords = open(os.path.join(pypath, "stopwords.txt")).readlines()
        stopwords = [stopword.strip() for stopword in stopwords]
        golds = [" ".join([word for word in ref.split() if word not in stopwords]) for ref in golds]
        predictions = [" ".join([word for word in hyp.split() if word not in stopwords]) for hyp in predictions]
    predictions = [str(i) + "\t" + pred.replace("\t", " ") for (i, pred) in enumerate(predictions)]
    golds = [str(i) + "\t" + gold.replace("\t", " ") for (i, gold) in enumerate(golds)]
    goldMap, predictionMap = computeMaps(predictions, golds)
    bleu = round(bleuFromMaps(goldMap, predictionMap)[0], 2)
    return bleu


def computeMaps(predictions, goldfile):
    predictionMap = {}
    goldMap = {}

    for row in predictions:
        cols = row.strip().split('\t')
        if len(cols) == 1:
            (rid, pred) = (cols[0], '')
        else:
            (rid, pred) = (cols[0], cols[1])
        predictionMap[rid] = [splitPuncts(pred.strip().lower())]

    for row in goldfile:
        (rid, pred) = row.split('\t')
        if rid in predictionMap:  
            if rid not in goldMap:
                goldMap[rid] = []
            goldMap[rid].append(splitPuncts(pred.strip().lower()))

    sys.stderr.write('Total: ' + str(len(goldMap)) + '\n')
    return (goldMap, predictionMap)

def bleuFromMaps(m1, m2):
    score = [0] * 5
    num = 0.0

    for key in m1:
        if key in m2:
            bl = bleu(m1[key], m2[key][0])
            score = [score[i] + bl[i] for i in range(0, len(bl))]
            num += 1
    return [s * 100.0000 / num for s in score]


import sys, math, re, xml.sax.saxutils
import subprocess
import os
import nltk
import json


def main():
    data = []
    with open('result.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            data.append(d)
    
    modified_data = []
    predictions = []
    references = []
    for item in data:  
        if item['security_type']=="No Issue":
            continue  
        pred_description = item.get('Description', '')
        pred_impact = item.get('Impact', '')
        pred_advice = item.get('Advice', '')
        pred_text = f"{pred_description} {pred_impact} {pred_advice}".strip()
        predictions.append(pred_text)
        ref_description = item.get('description', '')
        ref_impact = item.get('impact', '')
        ref_advice = item.get('advice', '')
        ref_text = f"{ref_description} {ref_impact} {ref_advice}".strip()
        references.append(ref_text)

    overall_bleu = bleu_fromstr(predictions, references, rmstop=False)
    print(f"Overall BLEU score: {overall_bleu}")

   

if __name__ == "__main__":
    main()
