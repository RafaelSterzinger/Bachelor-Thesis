#!/bin/sh
MAX=1
MIN=0
#DELTA=$(echo "($MAX - $MIN) / 10.0" | bc -l)
DELTA=0.1
for ((j = 0; j < 1; j++)); do
  echo "START WITH SEED $j"
  for ((i = 0; i < 9; i++)); do
    SECONDS=0
    echo "START GRIDSEARCH ITERATION $i"
    INT_COEFF=$(echo "$MAX - ($i*$DELTA)" | bc -l)
    EXT_COEFF=$(echo "$MIN + ($i*$DELTA)" | bc -l)
    echo $INT_COEFF
    echo $EXT_COEFF
    python ./src/run.py --ext_coeff $EXT_COEFF --int_coeff $INT_COEFF --seed 1
    echo "FINISH GRIDSEARCH ITERATION $i"
    echo "TIME ELAPSED $SECONDS"
  done
done
