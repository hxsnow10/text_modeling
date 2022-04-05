# encoding=utf-8
'''
sequence tagging(small) NER
'''
import os, sys
import tensorflow as tf
sys.path.append("..")
from utils.word2vec import getw2v
import json

default_batch_size = 100

 
import time
def now():
    return time.strftime("%Y-%m-%d-%H",time.localtime(time.time()))

def get_config(ceph_path="/ceph_ai", mode="train", branch="develop"):
    
    default_batch_size = 100
    
    zh_words_vec=getw2v( 
        vec_path=None,
        trainable=True,
        vocab_path="/ceph_ai/xiahong/data/segment_corpus/pos_tokens/train_vocab.txt",
        vocab_skip_head=False,
        max_vocab_size=500000,
        vec_size=108) # generate vocab, vocab_size, init_emb, vec_size
    
    class Config():
         
        class data_config():

            class pos_data():
                task_id = 1
                tags_paths=[ceph_path + "/xiahong/data/segment_corpus/pos_tokens/tags.txt"]
                train_paths=[ceph_path + "/xiahong/data/segment_corpus/pos_tokens/train.data"]
                dev_paths=[ceph_path + "/xiahong/data/segment_corpus/pos_tokens/test.data"]
                test_paths=[ceph_path + "/xiahong/data/segment_corpus/pos_tokens/test.data"]
                batch_size = default_batch_size 
                vocab = zh_words_vec.vocab
                sub_vocab= None
                tag_vocab = [{k: name.strip() for k,name in enumerate(open(path))} for path in tags_paths]
                tok="word_char"
                seq_len, sub_seq_len = 20, 5
                data_type="ner"
                names = ["input_zh_x", "input_zh_y_pos", "input_zh_x_length"]
                split = ' '
            
            task2configs = {"pos":pos_data}
            # train_sampling_args = 
         
        class model_config():

            class crf_args():
                uni_prob_shape=[len(zh_words_vec.vocab), 108]
                init_uni_prob=None
                pass

            class outputs_args_pos():
                objects="seq_tag"
                num_classes = 108# TODO
                use_crf=True
            
            class train_args():
                learning_method="adam_decay"
                start_learning_rate=0.01
                decay_steps = 6000
                decay_rate=0.8
                grad_clip=5
            
            placeholders = [
                ("input_zh_x", tf.int64, [None, None]),
                ("input_zh_x_length", tf.int64, [None]),
                ("input_zh_y_pos", tf.int64, [None, None]),
                ("dropout", tf.float32, None)
            ]
            
            net_crf = [# 这里输入输出的name表示self.name,而不是计算图中的名字
                [("input_zh_x", "input_zh_y_pos", "input_zh_x_length"),    "CRF",     crf_args,  "crf",   ("predictions_zh_pos", "loss_zh_pos")],
                [("loss_zh_pos",),         "TrainOp",        train_args,     "train2",   ("train_op_zh_pos","lr")],
            ]
            
            net_ = [# 这里输入输出的name表示self.name,而不是计算图中的名字
                [("input_zh_x", "input_zh_y_pos", "input_zh_x_length"),    "CRF",     crf_args,  "crf",   ("predictions_zh_pos", "loss_zh_pos")],
                [("loss_zh_pos",),         "TrainOp",        train_args,     "train2",   ("train_op_zh_pos","lr")],
            ]
            
            net = net_crf

            
            train_task2io={
                "pos":{"inputs":["input_zh_x", "input_zh_x_length", "input_zh_y_pos", "dropout"], "outputs":["loss_zh_pos", "train_op_zh_pos"] },
                }
            predict_task2io={
                "pos":{"inputs":["input_zh_x", "input_zh_x_length", "dropout"], "outputs": ["predictions_zh_pos","input_zh_x_length"] },
                }
        
        class train_config():
            task_type={"pos":"seq_tag"}
            epoch_num=20
            summary_steps=10
            session_conf = tf.ConfigProto(
                  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9),
                  device_count = {'CPU': 10, 'GPU':1},
                  allow_soft_placement=True,
                  log_device_placement=False)
            lrs = ["lr"] 
            start_learning_rate=0.01
            decay_steps = 2
            decay_rate=0.75
            ask_for_del=False
            super_params=['text_model', 'cnn_layer_num']
            z=locals()
            suffix='-'.join(["{}={}".format(name, z.get(name, None)) for name in super_params])
            tm=now()
            top_log_path = ceph_path+'/xiahong/LOG'
            model_dir=ceph_path+'/xiahong/RESULT/ner/{}/{}v2/model/{}'.format(branch, tm, suffix)
            # model_dir = '/tmp/xiahong/model0527'
            summary_dir=ceph_path+'/xiahong/RESULT/ner/{}/{}v2/log/{}'.format(branch, tm, suffix)
            model_path=os.path.join(model_dir,'model')
            print "MODEL_PATH=", model_path
            print "LOG_DIR=", summary_dir

    return Config

if __name__=="__main__":
    config = get_config()
    for name in dir(config):
        if name[0]=='_':continue
        print name, '\t', getattr(config, name)
        v = getattr(config, name)
        print '-'*40
        for name_2 in dir(v):
            print '\t', name_2, getattr(v,name_2)
