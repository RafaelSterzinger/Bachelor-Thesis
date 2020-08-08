#!/bin/sh
INT_COEFF=0
EXT_COEFF=1
SECONDS=0
for ((j = 0; j < 1; j++)); do
  echo "START WITH SEED $j"
  #python ./src/run.py --ext_coeff $EXT_COEFF --int_coeff $INT_COEFF --seed $j --feat_learning idf
  python ./src/run.py --ext_coeff $EXT_COEFF --int_coeff $INT_COEFF --seed $j --feat_learning none
  #python ./src/run.py --ext_coeff 0 --int_coeff 1 --seed $j --feat_learning none
  #python ./src/run.py --ext_coeff 0 --int_coeff 1 --seed $j --feat_learning idf
  #python ./src/run.py --ext_coeff 1 --int_coeff 0 --seed $j
  echo "END WITH SEED $j"
done
echo "TIME ELAPSED $SECONDS"
