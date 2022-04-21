#!/bin/bash
#coding=utf8

#根据clang_format，自动整理代码风格
# folder=test
folder=../text_modeling
exclude_folder=./source/C++/third_packages
format_files=`find "${folder}" -type f -regex ".*\.\(py\)" -not -path "./source/C++/third_packages/*" -not -path "./use/*" -not -path "./source/C++/build/*"`

# 增加版权声明
for file in ${format_files[@]};
do
    echo ${file}
    # pyformat ${file}
    autopep8 --in-place --aggressive --aggressive ${file}
    isort ${file}
done


