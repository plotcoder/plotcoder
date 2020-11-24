import argparse
import json
import os

scatter_word_list = ['scatter', "'scatter'", '"scatter"', 'scatter_kws', "'o'", "'bo'", "'r+'", '"o"', '"bo"', '"r+"']
hist_word_list = ['hist', "'hist'", '"hist"', 'bar', "'bar'", '"bar"', 'countplot', 'barplot']
pie_word_list = ['pie', "'pie'", '"pie"']
scatter_plot_word_list = ['lmplot', 'regplot']
hist_plot_word_list = ['distplot', 'kdeplot', 'contour']
normal_plot_word_list = ['plot']

reserved_words = scatter_word_list + hist_word_list + pie_word_list + scatter_plot_word_list + hist_plot_word_list + normal_plot_word_list


def preprocess(data_folder, init_data_name, prep_data_name, prep_hard_data_name=None, additional_samples=[],
               is_train=True):
    plot_samples = []
    clean_samples = []
    init_data_name = os.path.join(data_folder, init_data_name)
    with open(init_data_name, "r", encoding="utf-8") as fin:
        for i, line in enumerate(fin):
            sample = json.loads(line)

            # extract code sequence without comments and empty strings
            init_code_seq = sample['code_tokens']
            code_seq = []
            for tok in init_code_seq:
                if len(tok) == 0 or tok[0] == '#':
                    continue
                code_seq.append(tok)

            # filter out samples where 'plt' is not used
            while 'plt' in code_seq:
                pos = code_seq.index('plt')
                if pos < len(code_seq) - 1 and code_seq[pos + 1] == '.':
                    break
                code_seq = code_seq[pos + 1:]
            if not ('plt' in code_seq):
                continue

            plot_calls = []
            api_seq = sample['api_sequence']
            for api in api_seq:
                if api == 'subplot':
                    continue
                if api[-4:] == 'plot' and not ('_' in api):
                    plot_calls.append(api)

            exist_plot_calls = False
            for code_idx, tok in enumerate(code_seq):
                if not (tok in reserved_words + plot_calls):
                    continue
                if code_idx == len(code_seq) - 1 or code_seq[code_idx + 1] != '(':
                    continue
                exist_plot_calls = True
                break
            if not exist_plot_calls:
                continue

            url = sample['metadata']['path']
            if 'solution' in url.lower() or 'assignment' in url.lower():
                clean_samples.append(sample)
                if not is_train:
                    plot_samples.append(sample)
            else:
                plot_samples.append(sample)

    print('number of samples in the original partition: ', len(plot_samples))
    print('number of course-related samples in the partition: ', len(clean_samples))
    with open(os.path.join(data_folder, prep_data_name), 'w', encoding="utf-8") as writer:
        for sample in plot_samples:
            writer.write(json.dumps(sample) + "\n")
    if len(additional_samples) > 0:
        print('number of samples in the hard partition: ', len(additional_samples))
        with open(os.path.join(data_folder, prep_hard_data_name), 'w', encoding="utf-8") as writer:
            for sample in additional_samples:
                writer.write(json.dumps(sample) + "\n")
    return plot_samples, clean_samples


def main():
    arg_parser = argparse.ArgumentParser(description='JuiCe plot data extraction')
    arg_parser.add_argument('--data_folder', type=str, default='../data',
                            help="the folder where the datasets downloaded from the original JuiCe repo are stored. We will retrieve 'train.jsonl', 'dev.jsonl' and 'test.jsonl' here.")
    arg_parser.add_argument('--init_train_data_name', type=str, default='train.jsonl',
                            help="the filename of the original training data.")
    arg_parser.add_argument('--init_dev_data_name', type=str, default='dev.jsonl',
                            help="the filename of the original dev data.")
    arg_parser.add_argument('--init_test_data_name', type=str, default='test.jsonl',
                            help="the filename of the original test data.")
    arg_parser.add_argument('--prep_train_data_name', type=str, default='train_plot.jsonl',
                            help="the filename of the preprocessed training data. When set to None, it means that the training data is not preprocessed (this file is the most time-consuming for preprocessing).")
    arg_parser.add_argument('--prep_dev_data_name', type=str, default='dev_plot.jsonl',
                            help="the filename of the preprocessed dev data. When set to None, it means that the dev data is not preprocessed.")
    arg_parser.add_argument('--prep_test_data_name', type=str, default='test_plot.jsonl',
                            help="the filename of the preprocessed test data. When set to None, it means that the test data is not preprocessed.")
    arg_parser.add_argument('--prep_dev_hard_data_name', type=str, default='dev_plot_hard.jsonl',
                            help="the filename of the preprocessed hard split of the dev data. When set to None, it means that the dev data is not preprocessed.")
    arg_parser.add_argument('--prep_test_hard_data_name', type=str, default='test_plot_hard.jsonl',
                            help="the filename of the preprocessed hard split of the test data. When set to None, it means that the test data is not preprocessed.")

    args = arg_parser.parse_args()

    print('preprocessing training data:')
    train_plot_samples, train_plot_clean_samples = preprocess(args.data_folder, args.init_train_data_name,
                                                              args.prep_train_data_name, is_train=True)
    cnt_train_clean_samples = len(train_plot_clean_samples)

    print('preprocessing dev data:')
    dev_plot_samples, dev_plot_clean_samples = preprocess(args.data_folder, args.init_dev_data_name,
                                                          args.prep_dev_data_name,
                                                          prep_hard_data_name=args.prep_dev_hard_data_name,
                                                          additional_samples=train_plot_clean_samples[
                                                                             :cnt_train_clean_samples // 2],
                                                          is_train=False)

    print('preprocessing test data:')
    test_plot_samples, test_plot_clean_samples = preprocess(args.data_folder, args.init_test_data_name,
                                                            args.prep_test_data_name,
                                                            prep_hard_data_name=args.prep_test_hard_data_name,
                                                            additional_samples=train_plot_clean_samples[
                                                                               cnt_train_clean_samples // 2:],
                                                            is_train=False)


if __name__ == '__main__':
    main()
