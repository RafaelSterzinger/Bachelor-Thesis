#!/bin/sh
MAX=0.7
MIN=0.3
DELTA=$(echo "($MAX - $MIN) / 5.0" | bc -l)
for ((j = 0; j < 3; j++)); do
  echo "START WITH SEED $j"
  for ((i = 0; i < 11; i++)); do
    SECONDS=0
    echo "START GRIDSEARCH ITERATION $i"
    INT_COEFF=$(echo "$MAX - ($i*$DELTA)" | bc -l)
    EXT_COEFF=$(echo "$MIN + ($i*$DELTA)" | bc -l)
    python ./src/run.py --ext_coeff $EXT_COEFF --int_coeff $INT_COEFF --seed $j
    echo "FINISH GRIDSEARCH ITERATION $i"
    echo "TIME ELAPSED $SECONDS"
  done
done
