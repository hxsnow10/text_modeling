MULTITASK
=======

数据:
* 独立的postag数据  --> {input_x, x_length, input_y_1}
* 独立的ner数据   -->  {input_x, x_length, input_y_2}
* 情感分析数据 --> {input_x, x_length, input_y_3}
* 多语言

每个具体的数据文件生成器对每个sample对应 {name1:xx, name2: xx, ...}

while True:
    sample sample_data from train_data1, train_data2, ..
    sess.run(loss_k, train_op_k, sample_data)



