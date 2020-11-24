import json
import operator
import os

import fire


def add_token_to_dict(seq, vocab_dict):
    for tok in seq:
        if len(tok) == 0:
            continue
        if tok[0] == '#':
            continue
        if tok in vocab_dict:
            vocab_dict[tok] += 1
        else:
            vocab_dict[tok] = 1
    return vocab_dict


def build_vocab(
        data_folder: str,
        data_basename: str,
        code_freq_basename: str,
        code_vocab_basename: str,
        min_code_freq: int
):
    with open(os.path.join(data_folder, data_basename), "r", encoding="utf-8") as reader:
        samples = [json.loads(line) for line in reader]

    code_dict = {}
    for sample in samples:
        context = sample['context']
        for cell in context:
            if 'code_tokens' not in cell:
                continue
            code_context = cell['code_tokens']
            if type(code_context) != list:
                continue
            code_dict = add_token_to_dict(code_context, code_dict)
        code_dict = add_token_to_dict(sample['code_tokens'], code_dict)

    sorted_code_list = sorted(code_dict.items(), key=operator.itemgetter(1), reverse=True)
    print('Total number of code tokens (before filtering): ', len(sorted_code_list))

    with open(os.path.join(data_folder, code_freq_basename), "w", encoding="utf-8") as writer:
        json.dump(sorted_code_list, writer)

    # filter out rare tokens
    code_vocab = []

    for i, item in enumerate(sorted_code_list):
        if item[1] < min_code_freq:
            break
        code_vocab.append((item[0], i))

    print('Total number of code tokens (after filtering): ', len(code_vocab))
    with open(os.path.join(data_folder, code_vocab_basename), "w", encoding="utf-8") as writer:
        json.dump(code_vocab, writer)


if __name__ == '__main__':
    fire.Fire(build_vocab)
