#!/usr/bin/env python
# -*- encoding=utf8
#
#   Filename:
#   Author:         xiahong xiahahaha01@gmail.com
#   Create:

"""summary

description:

Usage:
foo = ClassFoo()
bar = foo.FunctionBar()
"""
# -*- encoding=utf8
#
#      Filename: graph.py
#
#        Author: xiahong
#   Description: ---
#        Create: 2021-08-25 01:12:16
# Last modified: 2021-08-25 16:44:41

# input:variables, each size batch*max_path_len
# output: score of path, size batch*1


class PathScore():

    def __init__(self):
        self.single_word_ratio = tf.Variable(0.1, trainable=True)
        self.tight_ratio = tf.Variable(0.25, trainable=True)
        self.cut_ratio = tf.Variable(1, trainable=True)

    def apply_by_formula1(self, inputs):
        path_len, word_len, q_count, t_count, qt_tight, tq_tight, qv, ngram_next_tight, ngram_next_match_count = inputs
        cnt = q_count * qt_tight + t_count * tq_tight
        tight = qt_tight + tq_tight
        single_word = tf.math.minimum(2 - word_len, 0)
        cnt_word_alpha = tf.ones_like(
            cnt) - single_word * (1 - self.single_word_ratio)
        cnt = cnt * cnt_word_alpha
        tight = (1 - single_word) * tight
        cnt = tf.max(tf.ones_like(cnt), cnt)
        global_weight = math.log(1.0 * cnt / N1) + self.tight_ratio * tight
        cut_weight = - tight_alpha * ngram_next_tight
        final_weight = tf.reduce_sum(
            global_weight + self.cut_ratio * cut_weight, -1)
        return final_weight


def PairwiseLoss(final_weight, postive_indexs, nagative_indexs, pair_weight):
    postive_weight = final_weight[postive_indexs]
    negative_weight = final_weight[nagative_indexs]
    loss = tf.reduce_sum(pair_weight * (postive_weight - negative_weight))
    return loss


class Model():
    def __init__(self):
        placeholders = [
            ("path_len", tf.int64, [None]),
            ("word_len", tf.int64, [None, None]),
            ("q_count", tf.int64, [None, None]),
            ("t_count", tf.float32, [None, None]),
            ("qt_tight", tf.float32, [None, None]),
            ("tq_tight", tf.float32, [None, None]),
            ("qv", tf.float32, [None, None]),
            ("ngram_next_tight", tf.float32, [None, None]),
            ("ngram_next_match_count", tf.float32, [None, None]),
            ("postive_indexs", tf.float32, [None]),
            ("nagative_indexs", tf.float32, [None]),
            ("pair_weight", tf.float32, [None, None]),
        ]
        for name, dtype, shape in placeholders:
            setattr(self, name, tf.placeholder(tf.int64, shape, name=name))
        self.path_inputs = [
            self.path_len,
            self.word_len,
            self.q_count,
            self.t_count,
            self.qt_tight,
            self.tq_tight,
            self.qv,
            self.ngram_next_tight,
            self.ngram_next_match_count]
        self.path_score_model = PathScore()
        self.final_weight = self.path_score_model.apply(self.path_inputs)
        self.loss = PairwiseLoss(
            self.final_weight,
            self.postive_indexs,
            self.nagative_indexs,
            self.pair_weight)


def train(model, train_data, dev_data, ):


'''
line_0: query, path_count, pair_count
line_k_path: attr=xxx,..\t...
...
line_k_pair: path_index, path_index
'''


class PathDataLoder():

    def __init__(self, attrs):
        self.attrs = attrs

    def load(input_path, batch_size):
        ii = open(input_path)
        path_len = []
        attrs = []
        pindexs = []
        nindexs = []
        weights = []
        while True:
            try:
                line = ii.readline()
                query, path_count, pair_count = line.rstrip().split('\t')
                path_count = int(path_count)
                pair_count = int(pair_count)
                for i in range(path_count):
                    line = ii.readline()
                    edges = line.split('\t')
                    edges = [dict([s.split('<=/>')
                                  for s in edge.split('<,/>')]) for edge in edges]
                    path_attrs = []
                    for edge in edges:
                        edge_attrs = [edge[attr] for attr in self.attrs]
                        path_attrs.append(attrs)
                    attrs.append(path_attrs)
                    path_len.append(len(edges))
                for i in range(pair_count):
                    line = ii.readline()
                    pindex, nindex, weight = line.split('\t')
                    pindexs.append(int(pindex))
                    nindexs.append(int(nindex))
                    weights.append(int(weight))
            except Exception, e:
                print e
                pass
            if len(path_len) >= batch_size:
                rval = {}
                rval["path_len"] = ap.array(path_len)
                rval["postive_indexs"] = pindexs
                rval["negative_indexs"] = nindexs
                rval["pair_weights"] = weights
                attrs_np = np.array(attrs)
                for k, arrr in enumerate(self.attrs)
                rval[attr] = attrs_np[:, :, k]
                yield rval
                path_len = []
                attrs = []
                pindexs = []
                nindexs = []
                weights = []


def train(
        sess,
        model,
        train_data,
        dev_datas=None,
        test_datas=None,
        summary_writers={},
        tags=None,
        config=None):
    oo = open(config.top_log_path, "a+")
    oo.write(now() + '\t' + config.model_dir + '\tstart_train\n')
    oo.close()
    best_dev, best_test, really_best_test = 0, 0, 0
    best_dev_metrics, best_test_metrics, really_best_test_metrics = None, None, None

    profiler = model_analyzer.Profiler(graph=sess.graph)
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    show_params(profiler)

    step = 0
    model.train_saver.save(sess, config.model_path, global_step=0)
    model.train_saver.save(sess, config.model_path)
    # score,dev_data_metrics = evaluate(sess,model,dev_data,tags)

    def eval(epoch, step, datas):
        for task_name, dev_data in datas.iteritems():
            oo = open(config.top_log_path, "a+")
            oo.write(
                '\t' +
                now() +
                '\t' +
                config.model_dir +
                '\tstart_eval' +
                str(epoch) +
                '\n')
            oo.close()
            if config.task_type[task_name] == "tag_exclusive":
                evaluate = evaluate_clf
            elif config.task_type[task_name] == "tag_noexclusive":
                evaluate = evaluate_tag
            elif config.task_type[task_name] == "seq_tag":
                evaluate = evaluate_seq_tag
            score, metrics = evaluate(
                sess, model, task_name, dev_data, target_names=tags[task_name])

            def add_summary(writer, metric, step):
                for name, value in metric.iteritems():
                    summary = tf.Summary(value=[
                        tf.Summary.Value(tag=name, simple_value=value),
                    ])
                    writer.add_summary(summary, global_step=step)
            add_summary(summary_writers[(task_name, 'dev')], metrics, step)
            model.train_saver.save(sess, config.model_path, global_step=step)
        return score, metrics
    for epoch in range(config.epoch_num):
        if epoch % config.decay_steps == 0:
            import math
            for lr in config.lrs:
                sess.run(
                    tf.assign(
                        getattr(
                            model,
                            lr),
                        config.start_learning_rate *
                        math.pow(
                            config.decay_rate,
                            epoch /
                            config.decay_steps)))

        for k, (task_name, inputs) in enumerate(train_data):
            inputs["dropout"] = 1.0
            # print "get in {}, task_name = {}".format(step, task_name)
            fd = {
                getattr(
                    model,
                    name): inputs[name] for name in model.train_task2io[task_name]["inputs"].keys()}
            out = model.train_task2io[task_name]["outputs"].values()
            if step % config.summary_steps != 0:
                out_v = sess.run(out, feed_dict=fd)
            else:
                out_v = sess.run(\
                    #out, feed_dict=fd,\
                    out + [model.step_summaries], feed_dict=fd,\
                    options=run_options,
                    run_metadata=run_metadata)
                summary_writers[(task_name, 'train')].add_summary(
                    out_v[-1], step)
                summary_writers[(task_name, 'train')].add_run_metadata(
                    run_metadata, "train" + str(step))
            # print "epoch={}\ttask={}\tstep={}\tglobal_step={}\tout_v={}".format(epoch, task_name, k, step ,out_v)
            step += 1
            if step > 0 and step % 10000 == 1:
                eval(epoch, step, dev_datas)

        if dev_datas:
            dev_f1, dev_metrics = eval(epoch, step, dev_datas)
            test_f1, test_metrics = eval(epoch, step, test_datas)
            if dev_f1 > best_dev:
                best_dev = dev_f1
                best_test = test_f1
                best_dev_metrics = dev_metrics
                best_test_metrics = test_metrics
            if test_f1 > really_best_test:
                really_best_test = test_f1
                really_best_test_metrics = test_metrics
            print '-' * 40
            print 'NOW BEST DEV = ', best_dev_metrics
            print 'AND TEST = ', best_test_metrics
            print 'AND REALLY TEST = ', really_best_test_metrics
            print '-' * 40

        else:
            model.train_saver.save(sess, config.model_path, global_step=step)


def main(mode):
    data_loder = PathDataLoder()
    train_data = data_loder.load(train_data_path)
    dev_data = data_loder.load(dev_data_path)
    with tf.Session(config=train_config.session_conf) as sess:
        # use tf.name_scope to manager variable_names
        model = Model()
        model.inits(sess, train_config.restore)
        model.save_info(train_config.model_dir)
        summary_writers = {
            (
                task_name,
                sub_path): tf.summary.FileWriter(
                os.path.join(
                    train_config.summary_dir,
                    task_name,
                    sub_path),
                sess.graph,
                flush_secs=5) for task_name in model_config.train_task2io for sub_path in [
                'train',
                'dev']}

        if mode == "train":
            train(sess, model,
                  data.train_data, data.dev_data, data.test_data,
                  summary_writers=summary_writers,
                  config=train_config, tags=data.tags)
        else:
            test(sess, model, data.dev_data, data.tags.vocab)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--config_path", default="./config.py")
    parser.add_argument("-r", "--restore", type=str, default="")
    parser.add_argument("-c", "--ceph_path", type=str, default="/ceph_ai")
    parser.add_argument("-b", "--branch", type=str, default="test")
    parser.add_argument("-m", "--mode", type=str, default="train")
    args = parser.parse_args()
    global config
    config = load_config(args.config_path, branch=args.branch)
    config.ceph_path = args.ceph_path
    config.train_config.restore = args.restore
    main(args.mode)
