# encoding=utf-8
import tensorflow as tf
from cnn import multi_filter_sizes_cnn_debug, multi_filter_sizes_cnn
from rnn import  rnn_func

import inspect
def get_func_args(depth=1):
    frame = inspect.currentframe(depth)
    args, name1, name2, values = inspect.getargvalues(frame)
    rval={}
    for d in [args, {}, {}]:
        for arg in d:
            rval[arg]=values.get(arg,None)
    return rval

class SubNet(object):

    def __init__(self, *args, **kwargs):
        pass

class Word2Vec(SubNet):

    def __init__(self, init_emb, w2v_shape, sub_init_emb, sub_w2v_shape, sub_cnn=None):
        args_=get_func_args()
        for arg in args_:
            setattr(self, arg, args_[arg])
        super(SubNet,self).__init__() 
        
    def __call__(self, input_x, input_x_sub=None):
        
        def build_emb(init_emb, name, shape=None):
            if init_emb is not None:
                init_emb=tf.constant(init_emb, dtype=tf.float32)
                W = tf.get_variable(name, initializer=init_emb, trainable=False)
            else:
                W = tf.get_variable(name, shape = shape, initializer = tf.contrib.layers.xavier_initializer(uniform = False))
            return W
        
        word_W=build_emb(self.init_emb, "word_W", self.w2v_shape)
        words_vec = tf.cast(tf.nn.embedding_lookup(word_W, input_x), tf.float32)
        shape_x = tf.shape(input_x)
        batch_size, seq_len = shape_x[0], shape_x[1]
        if self.sub_init_emb is not None  or self.sub_w2v_shape is not None:
            char_W=build_emb(self.sub_init_emb, "char_word_W", self.sub_w2v_shape)
            shape_x_sub = tf.shape(input_x_sub)
            char_len = shape_x_sub[-1]
            chars_vec = tf.cast(tf.nn.embedding_lookup(char_W, input_x_sub), tf.float32)
            chars_vec = tf.reshape(chars_vec,[batch_size*seq_len,char_len,-1])
            vec_size_c = self.sub_init_emb.shape[-1]
            word_vec= multi_filter_sizes_cnn(tf.expand_dims(chars_vec,-1), 10, vec_size_c, self.sub_cnn.char_filter_sizes, self.sub_cnn.char_filter_nums, name='char_cnn', reuse=False)
            cw_vec=tf.reshape(word_vec,[batch_size, seq_len, -1])
            words_vec=tf.concat([words_vec, cw_vec], -1)
            words_vec.set_shape([None, None, 50+40])
        return words_vec

class Cnn(SubNet):
    def __init__(self, filter_sizes, filter_nums, cnn_layer_num, gated, bi=True, debug=False):
        args_=get_func_args()
        for arg in args_:
            setattr(self, arg, args_[arg])
        super(Cnn,self).__init__()

    def __call_(self, words_vec):
        shape_x = tf.shape(words_vec)
        batch_size, seq_len, vec_size = shape_x[0], shape_x[1], shape_x[2]
        if self.debug:
            self.sent_vec, self.pooled_index = multi_filter_sizes_cnn_debug(words_emb, self.seq_len, self.vec_size, self.filter_sizes, self.filter_nums, name='cnn', reuse=self.reuse)
        else:
            for i in range(self.cnn_layer_num):
                words_vec = tf.expand_dims(words_vec, -1)
                words_vec_1 = multi_filter_sizes_cnn(words_vec, seq_len, vec_size, self.filter_sizes, self.filter_nums, name='cnn'+str(i)+'_forward', pooling=False, reuse=self.reuse, gated=self.gated, padding="same", front_pad='MAX')
                words_vec_2 = multi_filter_sizes_cnn(words_vec, seq_len, vec_size, self.filter_sizes, self.filter_nums, name='cnn'+str(i)+'_backward', pooling=False, reuse=self.reuse, gated=self.gated, padding="same", front_pad=0)
                words_vec = tf.concat([words_vec_1, words_vec_2], 2) if self.bi else words_vec_1
                vec_size = 2* sum(self.filter_nums) if self.bi else sum(self.filter_nums)
        return words_vec
    
'''
    def build_sent_add(self):
        self.mask = tf.cast(tf.sequence_mask(self.input_sequence_length, config.seq_len), tf.float32)
        self.sent_vec = tf.reduce_sum(tf.expand_dims(self.mask,-1)*self.words_emb[:,:,:,0],1)
        self.sent_vec = tf.nn.l2_normalize(self.sent_vec, dim = -1)
        
    def build_sent_add_idf(self):
        self.mask = tf.cast(tf.sequence_mask(self.input_sequence_length, config.seq_len), tf.float32)
        self.idf_x = tf.nn.embedding_lookup(self.idf, self.input_x)
        self.sent_vec = tf.reduce_sum(tf.expand_dims(self.mask,-1)*self.words_emb[:,:,:,0]\
                    *tf.expand_dims(self.idf_x,-1),1)
        self.sent_vec = tf.nn.l2_normalize(self.sent_vec, dim = -1)

    def build_hs_atta(self):
        pass
'''
class SumConcat(SubNet):
    def __init__(self):
        pass
    
    def move(self, words_vec, i):
        if i>0:
            return tf.pad(words_vec[:, i:], [[0,0],[0,i],[0,0]])
        else:
            return tf.pad(words_vec[:, :i], [[0,0],[-i,0],[0,0]])
            
    def __call__(self, words_vec, input_sequence_length, dropout=None):
        words_vec_0 = words_vec
        words_vec_p1 = self.move(words_vec,1)
        words_vec_p2 = self.move(words_vec,2)
        words_vec_n1 = self.move(words_vec,-1)
        words_vec_n2 = self.move(words_vec,-2)

        words_vec_1 = words_vec_0+words_vec_p1
        words_vec_2 = words_vec_1+words_vec_p2
        words_vec_3 = words_vec_0+words_vec_n1
        words_vec_4 = words_vec_3+words_vec_n2

        words_vec = tf.concat([words_vec_0, words_vec_1, words_vec_2, words_vec_3, words_vec_4], 2)
        print words_vec_0.get_shape()
        print words_vec.get_shape()
        return words_vec
        
class Rnn(SubNet):
    def __init__(self, rnn_cell, cell_size, rnn_layer_num, attn_type, bi):
        args_=get_func_args()
        for arg in args_:
            setattr(self, arg, args_[arg])
        super(Rnn,self).__init__()

    def __call__(self, words_vec, input_sequence_length, dropout=None):
        words_vec=rnn_func(words_vec, self.rnn_cell, self.cell_size, self.rnn_layer_num, 
            sequence_length=input_sequence_length, attn_type=self.attn_type, bi=self.bi)
        return words_vec
'''
    def build_cnn_rnn(self):
        words_emb= multi_filter_sizes_cnn(self.words_emb, self.seq_len, self.vec_size, config.filter_sizes, config.filter_nums, name='cnn', reuse=False, pooling=False, padding="same")
        words_emb=rnn_func(words_emb, config.rnn_cell, config.cell_size, config.rnn_layer_num, sequence_length=self.input_sequence_length,attn_type=config.attn_type, bi=config.bi)

    def build_graph_cnn(self):
        several ways to do it:
        * make batch=1, generate sparse/dense W=[N*b,N*D] by Conv and graph link
          niubility: use tf.nn.embedding_lookup_sparse(Conv, sparse_L_L)
        * reshape to place indexs with same context_size, and reshape back. 
            could do with batch>1, but code some difficult
        implement sparse first.
        pass 
    def build_text_repr(self):
            if config.objects=="tag":
                if config.text_model in ['cnn', 'gated-cnn','rnn-cnn']:
                    self.sent_vec=tf.reduce_max(self.words_emb,1)
                else:
                    self.sent_vec=self.words_emb[:,-1,:]
'''
        
class ExclusiveTagOutputs(SubNet):

    def __init__(self, debug, l2_lambda, class_weights):
        args_=get_func_args()
        for arg in args_:
            setattr(self, arg, args_[arg])
        super(ExclusiveOutputs,self).__init__()

    def __call__(self, inputs, input_y):
        with tf.name_scope("output"):
            scores = tf.layers.dense(inputs, num_classes, name="dense")
            predictions = tf.argmax(scores, 1, name="predictions")
            # if self.debug:
            #    inputs2=tf.stack([inputs,]*self.num_classes,-1)
            #    self.scores2 = tf.multiply(inputs2, tf.expand_dims(W,0))

        with tf.name_scope("loss"):

            losses = tf.nn.softmax_cross_entropy_with_logits(labels=input_y, logits=scores)
            if self.class_weights:
                class_weights = self.class_weights#这里做了个糟糕的假设，假设input_y[i]只有一个是1，其余是0
                weights = tf.reduce_sum(class_weights * tf.cast(input_y,tf.float32),axis=1)
                losses = losses * weights
             
            l2_loss = tf.add_n([ tf.cast(tf.nn.l2_loss(v), tf.float32) for v in tf.trainable_variables() if 'bias' not in v.name ])
            loss = tf.reduce_mean(losses) + config.l2_lambda * l2_loss
        tf.summary.scalar("loss", loss)    
        return predictions, loss


        # Accuracy
        '''
        with tf.name_scope("metrics"):
            correct_predictions = tf.equal(self.predictions, self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
        self.outputs.append(correct_predictions)
        '''
        # tf.summary.scalar("accuracy", self.accuracy)    

class NoExclusiveTagOutputs(SubNet):

    def __init__(self, ):
        args_=get_func_args()
        for arg in args_:
            setattr(self, arg, args_[arg])
        pass

    def __call__(self, input_y):
        '''有几种方法来处理不互斥的tags
        '''
        with tf.name_scope("output"):
            with tf.variable_scope("share"):
                scores = tf.layers.dense(inputs, self.num_classes*2, name="dense", reuse=self.reuse)
            scores = tf.reshape(scores, [self.batch_size, num_classes, 2])
            predictions = tf.argmax(scores, -1, name="predictions")

        with tf.name_scope("loss"):
            input_y = tf.one_hot(input_y,depth=2,axis=-1)
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=input_y)
            if config.use_label_weights:
                losses = losses*tf.expand_dims(self.class_weights,0)
            loss = tf.reduce_mean(losses)
        tf.summary.scalar("loss", loss)
        '''
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            self.class_accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), axis = 0, name="class_accuracy")
        '''
        return predictions, loss
'''
    def build_weighted_sampled_outputs(self, inputs, num_classes):
        with tf.name_scope("output"):
            weights = tf.get_variable("weights",shape=[num_classes, self.repr_size],dtype=tf.float32)
            biases = tf.get_variable("biases",shape=[num_classes], dtype=tf.float32)
            tf.summary.histogram('weights',weights)
            tf.summary.histogram('biases',biases)

        # as class number is big, use sampled softmax instead dense layer+softmax
        with tf.name_scope("loss"):
            tags_prob = tf.pad(self.input_y_prob,[[0,0],[0,config.num_sampled]])
            out_logits, out_labels= _compute_sampled_logits( weights, biases, self.input_y, inputs,\
                    config.num_sampled, num_classes, num_true= config.max_tags )
            # TODO:check out_labels keep order with inpuy
            weighted_out_labels = out_labels * tags_prob*config.max_tags
            # self.out_labels = weighted_out_labels
            self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=out_logits, labels=weighted_out_labels))
        
        with tf.name_scope("outputs"):
            logits = tf.nn.softmax(tf.matmul(inputs, tf.transpose(weights)) + biases)
            self.output_values, self.ouput_indexs = tf.nn.top_k(logits, config.topn)

        with tf.name_scope("score"):
            self.scores = self.loss/tf.cast(self.batch_size, tf.float32)
            #self.accuracy = tf.reduce_sum( self.top_prob )

        tf.summary.scalar('loss', self.loss)

    def build_sampled_outputs(self, inputs, num_classes, exclusive):
        weights = tf.get_variable("weights",shape=[num_classes, self.repr_size],dtype=tf.float32)
        biases = tf.get_variable("biases",shape=[num_classes], dtype=tf.float32)
        if config.tag_exclusive:
            losses=tf.nn.sampled_softmax_loss(weights, biases, self.input_y, inputs, config.num_sampled, num_classes, num_true=1, partition_strategy="div")
        else:
            losses=tf.nn.nec_loss(weights, biases, self.input_y, inputs, config.num_sampled, num_classes, num_true=1, partition_strategy="div")
        self.loss=tf.reduce_sum(losses)
        self.scores= tf.nn.bias_add(tf.matmul(inputs, tf.transpose(weights)), biases)
        self.output_values, self.ouput_indexs = tf.nn.top_k(self.scores, 2)
'''
class CRF(SubNet):
    
    def __init__(self, ngram_sizes=[], tag_size=None, init_embs=[]):
        args_=get_func_args()
        for arg in args_:
            setattr(self, arg, args_[arg])
        super(CRF,self).__init__()

    def __call__(self, input_x, input_y, input_sequence_length):
        
        def build_emb(init_emb, name, shape=None, trainable=True):
            if init_emb is not None:
                init_emb=tf.constant(init_emb, dtype=tf.float32)
                W = tf.get_variable(name, initializer=init_emb, trainable=False)
            else:
                W = tf.get_variable(name, shape = shape, initializer = tf.contrib.layers.xavier_initializer(uniform = True), dtype=tf.float32)
            return W
         
        print self.ngram_sizes, self.tag_size
        W1 = [ build_emb(None, "prob_{}".format(i), [ngram_size, self.tag_size], True) for i,ngram_size in enumerate(self.ngram_sizes) ] 
        scoress1 = [tf.nn.embedding_lookup(W1[i], input_x[:,:,i]) for i in range(len(W1))]
        scores = tf.cast(tf.add_n(scoress1), tf.float32)
        
        W2 = [ build_emb(self.init_embs[i], "feature_{}".format(i), None, False) for i,ngram_size in enumerate(self.ngram_sizes) ]
        scoress2 = [tf.layers.dense(tf.nn.embedding_lookup(W2[i], input_x[:,:,i]), self.tag_size, name="linear_map_{}".format(i))  for i in range(len(W2))]
        # scoress2 = [tf.layers.dense(scoress2[i], self.tag_size, name="linear_map_2_{}".format(i), activation=tf.nn.relu)  for i in range(len(scoress2))]

        # scores = scores+tf.cast(tf.add_n(scoress2), tf.float32)
        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
            scores, tf.cast(input_y, tf.int32), input_sequence_length)
        loss = tf.reduce_mean(-log_likelihood)
        # l2_loss = tf.add_n([ tf.cast(tf.nn.l2_loss(v), tf.float32) for v in tf.trainable_variables() if 'bias' not in v.name ])
        # loss = loss + 0.001 * l2_loss
        predictions, _=tf.contrib.crf.crf_decode( scores, transition_params, input_sequence_length)
        return predictions, loss 

class SeqTagOutputs(SubNet):

    def __init__(self, num_classes, use_crf=True):
        args_=get_func_args()
        for arg in args_:
            setattr(self, arg, args_[arg])
        self.num_classes = num_classes
        self.use_crf=use_crf

    def __call__(self, inputs, input_y, input_sequence_length):
        print inputs
        batch_size=tf.shape(inputs)[0]
        seq_len=tf.shape(inputs)[1]
        vec_size = inputs.get_shape().as_list()[2] or tf.shape(inputs)[2]
        inputs = tf.reshape(inputs, [batch_size*seq_len, vec_size])
        print inputs 
        scores = tf.layers.dense(inputs, self.num_classes, name="dense")
        scores = tf.reshape(scores, [batch_size, seq_len, self.num_classes])
        # bug when replace config.seq_len with ntime_steps
        if self.use_crf:
            log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
            scores, input_y, input_sequence_length)
            loss = tf.reduce_mean(-log_likelihood)
            predictions, _=tf.contrib.crf.crf_decode( scores, transition_params, input_sequence_length)
        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores, labels=input_y)
            mask = tf.sequence_mask(input_sequence_length, seq_len)
            losses = tf.boolean_mask(losses, mask)
            loss = tf.reduce_mean(losses)
            predictions=tf.argmax(scores,-1)

        print tf.summary.scalar("loss", loss)

        return predictions, loss

    def build_seq_sampled_tag_outputs(self):
        pass
        
        # consider seq2seq, encoder-decoder

def Outputs(objects, num_classes, *args, **kwargs):
    if objects=="tag":
        if config.sampled:
            self.build_sampled_outputs(self.sent_vec, self.num_classes, config.tag_exclusive)
        elif config.tag_exclusive:
            ExclusiveOutputs(self.num_classes)
        else:
            NonExclusiveOutputs(self.num_classes)
    elif objects=="seq_tag":
        return SeqTagOutputs(num_classes, *args, **kwargs)

class TrainOp(SubNet):

    def __init__(self, learning_method, start_learning_rate, decay_steps, decay_rate, grad_clip):
        args_=get_func_args()
        for arg in args_:
            setattr(self, arg, args_[arg])
        super(TrainOp,self).__init__()

    def __call__(self, loss):
        global_step = tf.Variable(0, trainable=False) 
        # self.learning_rate = tf.train.exponential_decay(self.start_learning_rate, global_step,
        #        self.decay_steps, self.decay_rate, staircase=True)
        # Passing global_step to minimize() will increment it at each step.
        self.learning_rate = tf.Variable(0.003, name="learning_rate")
        tf.summary.scalar("learning_rate", self.learning_rate)
        
        if self.learning_method=='adam':
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
        elif self.learning_method=='adam_decay':
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
        elif self.learning_method=='sgd':
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        elif self.learning_method == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        elif self.learning_method == 'pro':
            optimizer = tf.train.ProximalGradientDescentOptimizer(self.learning_rate)
        if self.grad_clip: 
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars),
                    self.grad_clip)
            train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)
        else:
            train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op, self.learning_rate

subnets={
    "Word2Vec":Word2Vec,
    "Cnn":Cnn,
    "Rnn":Rnn,
    "Outputs":Outputs,
    "TrainOp":TrainOp,
    "SumConcat":SumConcat,
    "CRF":CRF
    }
print subnets
