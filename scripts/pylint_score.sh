#!/usr/bin/env sh

######################################################################
# @author      : xiahong (xiahahaha01@gmail.com)
# @file        : pylint_score
# @created     : 星期六  4 09, 2022 22:27:05 CST
#
# @description : 
######################################################################
echo $1
pylint --variable-rgx="^[a-z][a-z0-9]*((_[a-z0-9]+)*)?$"\
    --argument-rgx="^[a-z][a-z0-9]*((_[a-z0-9]+)*)?$"\
    --const-rgx="[a-z_][a-z0-9_]{2,30}$" $1

