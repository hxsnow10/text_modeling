# encoding=utf-8
import sys
sys.path.append('..')
from tf_utils.data import LineBasedDataset, conll_lines_processing
from itertools import islice

def load_data_ner(config, mode="train"):
    '''every line like: tok[\ttag1..]\n or blackline
    specifically, LineBasedDataset, use '\n\n' as split
    '''
    class Data():
        words=config.vocab 
        sub_words= config.sub_vocab
        tagss= config.tag_vocab
        print tagss[0]
        seq_p=lines_processing=conll_lines_processing(config.split, tagss, config.seq_len, sub_words=sub_words, sub_seq_len = config.sub_seq_len, 
            ngram_defs=config.ngram_defs, ngram_words=config.ngram_words)
        tags=lines_processing.tag_vocabs[0]
        print 'LEN TAGS= ', len(tags)
        train_data=LineBasedDataset(config.train_paths, line_processing=lines_processing, batch_size=config.batch_size, split='\n\n', names= config.names)
        dev_data=LineBasedDataset(config.dev_paths, line_processing=lines_processing, batch_size=config.batch_size, split='\n\n', names = config.names)
        test_data=LineBasedDataset(config.test_paths, line_processing=lines_processing, batch_size=config.batch_size, split='\n\n', names = config.names)
        class_weights=None
        object="seq_tag"
    return Data 
