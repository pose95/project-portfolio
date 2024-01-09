# -*- coding: utf-8 -*-
"""
Created on Tue May 16 18:05:37 2023

@author: matteo posenato
"""
def preprocess_data(train_infile, test_infile, output_dir, vocab_size, log_transform=False):
    print("Loading Spacy")
    parser = English()

    with codecs.open(train_infile, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    with codecs.open(test_infile, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    train_X, train_vocab, train_indices, train_y, word_freqs, label_list, train_X_indices, train_delect_index = load_and_process_data(
        train_infile, vocab_size, parser, log_transform=log_transform)
    test_X, _, test_indices, test_y, _, _, test_X_indices, test_delect_index = load_and_process_data(
        test_infile,vocab_size, parser,vocab=train_vocab,log_transform=log_transform,label_list=label_list)
    fh.save_sparse(train_X, os.path.join(output_dir, 'train.npz'))
    fh.write_to_json(train_vocab, os.path.join(output_dir, 'train.vocab.json'))
    fh.write_to_json(train_indices, os.path.join(output_dir, 'train.indices.json'))
    fh.save_sparse(train_y, os.path.join(output_dir, 'train.labels.npz'))
    fh.save_sparse(train_X_indices, os.path.join(output_dir, 'train_X_indices.npz'))
    fh.write_list_to_text(train_delect_index, os.path.join(output_dir, 'train_delect_index.txt'))

    fh.save_sparse(test_X, os.path.join(output_dir, 'test.npz'))
    fh.write_to_json(test_indices, os.path.join(output_dir, 'test.indices.json'))
    fh.save_sparse(test_y, os.path.join(output_dir, 'test.labels.npz'))
    fh.save_sparse(test_X_indices, os.path.join(output_dir, 'test_X_indices.npz'))
    fh.write_list_to_text(test_delect_index, os.path.join(output_dir, 'test_delect_index.txt'))

    n_labels = len(label_list)
    label_dict = dict(zip(range(n_labels), label_list))
    fh.write_to_json(label_dict, os.path.join(output_dir, 'train.label_list.json'))
    fh.write_to_json(list(word_freqs.tolist()), os.path.join(output_dir, 'train.word_freq.json'))
