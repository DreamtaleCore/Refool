#!/bin/bash

#for((i=1;i<10;i++));
#do
#echo Infection rate: 0.$i, Adaptive;
#python3 train.py --output_path checkpoints-tmp/adaptive-0.$i --attack_postfix _with_atk-adaptive- -infect_rate 0.$i
#done
#
#for((i=1;i<10;i++));
#do
#echo Infection rate: 0.$i, Manual;
#python3 train.py --output_path checkpoints-tmp/manaul-0.$i --attack_postfix _with_atk-manual- -infect_rate 0.$i
#done

echo Training 8-manual
python3 train.py --output_path checkpoints-invisible --attack_postfix _with_atk-8-manual- --infect_rate 0.9
echo Done 8-manual

echo Training 14-manual
python3 train.py --output_path checkpoints-invisible --attack_postfix _with_atk-14-manual- --infect_rate 0.9
echo Done 14-manual

echo Training 20-manual
python3 train.py --output_path checkpoints-invisible --attack_postfix _with_atk-20-manual- --infect_rate 0.9
echo Done 20-manual

echo Training 35-manual
python3 train.py --output_path checkpoints-invisible --attack_postfix _with_atk-35-manual- --infect_rate 0.9
echo Done 35-manual

echo Training 39-manual
python3 train.py --output_path checkpoints-invisible --attack_postfix _with_atk-39-manual- --infect_rate 0.9
echo Done 39-manual

