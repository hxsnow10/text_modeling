#!/usr/bin/env python
# -*- encoding=utf8
#
#   Filename:       normalization
#   Author:         xiahong xiahahaha01@gmail.com
#   Create:         19/04/2022
#   Description:    ---
"""one line of summary

tf.nn.batch_normalization
tf.keras.layers.Normalization(
    axis=-1, mean=None, variance=None, **kwargs
)
__call__(inputs, training =  True/False)

tf.keras.layers.BatchNormalization
tf.keras.layers.LayerNormalization
tf.keras.layers.InstanceNormalization
tfa.layers.WeightNormalization
"""

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--config_path", default="./config.py")
    args = parser.parse_args()
    main(args.config_path)

