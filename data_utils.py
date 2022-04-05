# encoding=utf-8
import argparse
from data_loaders import *
from random import randint

def load_data_(config):
    if config.data_type=="ner":
        return load_data_ner(config)
    elif config.data_type=="w2v":
        return load_data_w2v(config)
    else:
        return load_data_clf(config) 

def load_data(config):
    class sampling():
    
        def __init__(self,task2data):
            self.task2data = task2data
            print self.task2data

        def __iter__(self):
            todo = [[name,data.iter()] for name,data in self.task2data.iteritems()]
            while True:
                k = randint(0,len(todo)-1)
                try:
                    yield todo[k][0], todo[k][1].next()
                except Exception,e:
                    print e
                    import traceback
                    traceback.print_exc()
                    todo.pop(k)
                    if not todo: break
    class data():
        task2data={}
        for task_name,task_config in config.task2configs.iteritems():
            task2data[task_name] = load_data_(task_config)
        print task2data
        train_data = sampling({name:data_.train_data for name,data_ in task2data.iteritems()})
        dev_data = {name:data_.dev_data for name,data_ in task2data.iteritems()}
        test_data = {name:data_.test_data for name,data_ in task2data.iteritems()}
        tags = {name:data_.tags.vocab for name,data_ in task2data.iteritems()}
        
    return data
 
if __name__=="__main__":
    from tf_utils import load_config
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--config_path", default=".")
    args = parser.parse_args()
    config=load_config(args.config_path)
    data=load_data(config.data_config)
    for data_ in [data.train_data]:
        for k,(task_name, inputs) in enumerate(data_):
            print '-'*20,'batch ',k,'-'*20, task_name
            for name,inp in inputs.iteritems():
                print name
                # print inp[:]
                print inp.shape, inp.dtype
                try:
                    print inp[0][:100]
                except:
                    print inp
                raw_input('XXXXXXXXXXXXX')
    for data_ in data.dev_data.values():
        print data_
        for k,inputs in enumerate(data_):
            raw_input('XXXXXXXXXXXXX')
            for name,inp in inputs.iteritems():
                print name
                print inp[:]
                print inp.shape, inp.dtype
            
            # if k>=20:break

