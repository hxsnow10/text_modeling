#encoding=utf-8
# import gensim
import os, sys
import numpy as np    
from collections import OrderedDict
from itertools import islice
UNK='</s>'
PAD='</pad>'

class gensim_model():
    
    def __init__(self, model_path):
        model = gensim.models.Word2Vec.load_word2vec_format(model_path, binary=False)

    def most_similar(self, word):
        pass

def normalized(a, axis=-1, order=2):
    """Utility function to normalize the rows of a numpy array."""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1 
    return a / np.expand_dims(l2, axis)

def load_w2v(w2v_path, dtype=np.float16, norm=False, max_vocab_size=None, limited_words=[]):
    limited_words=set(limited_words)
    ii=open(w2v_path,'r')
    n,l=None,None
    try:
        n,l=ii.readline().strip().split()
        n,l=int(n),int(l)
    except:
        pass
    w2v=OrderedDict()
    w2v.dim=l
    for k,line in enumerate(ii):
        try: 
            wl=line[:-1].split()
            word=wl[0]
            if limited_words and word not in limited_words:continue
            value=[float(x) for x in wl[1:]]
            assert len(value)==l or not l
            value=np.array(value, dtype=dtype)
            if norm:
                value=normalized(value)[0]
            w2v[word]=value
        except:
            print line
        if max_vocab_size and k>=max_vocab_size:break

    return w2v

def save_w2v(w2v, w2v_path):
    oo=open(w2v_path,'w')
    l=0
    for x in w2v:
        l=w2v[x].shape[0]
        break
    n=len(w2v)
    oo.write(str(n)+' '+str(l)+'\n')
    for w,v in w2v.iteritems():
        oo.write(' '.join([w]+[str(x) for x in list(v)])+'\n')
   
class getw2v(object):
    def __init__(self, vec_path, trainable,
        vocab_path, vocab_skip_head, max_vocab_size, vec_size=300, norm=False):
        self.vec_path=vec_path
        self.trainable=trainable
        self.vocab_path=vocab_path
        self.vocab_skip_head=vocab_skip_head
        self.max_vocab_size=max_vocab_size
        self.vec_size=int(open(vec_path).readline().strip().split()[-1])\
            if vec_path else vec_size
        self.norm=norm
        ff=open(vocab_path)
        if vocab_skip_head:
            ff.readline()
        vocab={}
        for k,word in enumerate(islice(ff,max_vocab_size)):
            if k>=max_vocab_size: break
            if word.split() and (k<max_vocab_size or max_vocab_size is None):
                vocab[word.split()[0]]=len(vocab)
        if UNK not in vocab:
            k=len(vocab)
            vocab[UNK]=k
            print UNK, k
        if PAD not in vocab:
            k=len(vocab)
            vocab[PAD]=k
            print PAD, k
        self.vocab = {k:name for name,k in vocab.iteritems()}
        self.vocab_size = len(self.vocab)
        print "VEC INFO ", self.vocab_size, self.vec_size
        if vec_path:
            if not os.path.exists(vec_path):
                raise Exception("File {} not exists".format(vec_path))
            w2v = load_w2v(vec_path, max_vocab_size=max_vocab_size, norm=norm)
            self.init_emb = np.array(w2v.values()+[[0,]*self.vec_size,]*(len(self.vocab)-len(w2v)), dtype=np.float32)
        else:
            self.init_emb = np.random.random([len(self.vocab),self.vec_size])*0.01
