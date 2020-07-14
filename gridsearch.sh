#!/bin/sh
MAX=1
MIN=0
DELTA=$(echo "($MAX - $MIN) / 10.0" | bc -l)
for ((i = 0; i < 11; i++)); do
  SECONDS=0
  echo "START GRIDSEARCH ITERATION $i"
  INT_COEFF=$(echo "$MAX - ($i*$DELTA)" | bc -l)
  EXT_COEFF=$(echo "$MIN + ($i*$DELTA)" | bc -l)
  python ./src/run.py --ext_coeff $EXT_COEFF --int_coeff $INT_COEFF

  echo "TIME ELAPSED $SECONDS"
  echo "FINISH GRIDSEARCH ITERATION $i"
done
