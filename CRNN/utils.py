from collections import defaultdict

import torch
import numpy as np
from scipy.special import logsumexp


class CharEncoder():
    def __init__(self, charset):
        self.char_code = {}
        self.char_code[' '] = 0
        for i, char in enumerate(charset):
            self.char_code[char] = i + 1
    
    def __call__(self, text):
        encoded_text = [self.char_code[c] for c in text]
        return encoded_text
    
def encode_char(charset, text):
    charset = ' ' + charset
    encoded_text = [charset.index(c) for c in text]
    return encoded_text
    
def decode_char(charset, encoded_text):
    charset = ' ' + charset
    char_list = [charset[i] for i in encoded_text]
    return char_list
    
def get_charset(path='charset.txt'):
    with open(path) as f:
        charset = f.read()
    
    return charset

DEFAULT_EMISSION_THRESHOLD = 0.01
NINF = -1 * float('inf')
def _reconstruct(labels, blank=0):
    new_labels = []
    # merge same labels
    previous = None
    for l in labels:
        if l != previous:
            new_labels.append(l)
            previous = l
    # delete blank
    new_labels = [l for l in new_labels if l != blank]

    return new_labels

def beam_search_decode(emission_log_prob, blank=0, **kwargs):
    beam_size = kwargs['beam_size']
    emission_threshold = kwargs.get('emission_threshold', np.log(DEFAULT_EMISSION_THRESHOLD))

    length, class_count = emission_log_prob.shape

    beams = [([], 0)]  # (prefix, accumulated_log_prob)
    for t in range(length):
        new_beams = []
        for prefix, accumulated_log_prob in beams:
            for c in range(class_count):
                log_prob = emission_log_prob[t, c]
                if log_prob < emission_threshold:
                    continue
                new_prefix = prefix + [c]
                # log(p1 * p2) = log_p1 + log_p2
                new_accu_log_prob = accumulated_log_prob + log_prob
                new_beams.append((new_prefix, new_accu_log_prob))

        # sorted by accumulated_log_prob
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_size]

    # sum up beams to produce labels
    total_accu_log_prob = {}
    for prefix, accu_log_prob in beams:
        labels = tuple(_reconstruct(prefix, blank))
        # log(p1 + p2) = logsumexp([log_p1, log_p2])
        total_accu_log_prob[labels] = \
            logsumexp([accu_log_prob, total_accu_log_prob.get(labels, NINF)])

    labels_beams = [(list(labels), accu_log_prob)
                    for labels, accu_log_prob in total_accu_log_prob.items()]
    labels_beams.sort(key=lambda x: x[1], reverse=True)
    labels = labels_beams[0][0]

    return labels


def ctc_decode(log_probs, charset, decode_char=None, blank=0, method='beam_search', beam_size=10):
    emission_log_probs = np.transpose(log_probs.cpu().numpy(), (1, 0, 2))
    # size of emission_log_probs: (batch, length, class)

    decoders = {
        # 'greedy': greedy_decode,
        'beam_search': beam_search_decode,
        # 'prefix_beam_search': prefix_beam_decode,
    }
    decoder = decoders[method]

    decoded_list = []
    for emission_log_prob in emission_log_probs:
        decoded = decoder(emission_log_prob, blank=blank, beam_size=beam_size)
        if decode_char:
            decoded = decode_char(charset, decoded)
        decoded_list.append(decoded)
    return decoded_list
    