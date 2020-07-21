#!/bin/sh
MAX=0.7
MIN=0.56
RUNS=5
SEEDS=1
DELTA=$(echo "($MAX - $MIN) / $RUNS" | bc -l)
for ((j = 0; j < $SEEDS; j++)); do
  echo "START WITH SEED $j"
  for ((i = 1; i <= $RUNS; i++)); do
    SECONDS=0
    echo "START GRIDSEARCH ITERATION $i"
    EXT_COEFF=$(echo "$MIN + (($i-0.5) * $DELTA)" | bc -l)
    INT_COEFF=$(echo "1-$EXT_COEFF" | bc -l)
    python ./src/run.py --ext_coeff $EXT_COEFF --int_coeff $INT_COEFF --seed $j
    echo $EXT_COEFF
    echo $INT_COEFF
    echo "FINISH GRIDSEARCH ITERATION $i"
    echo "TIME ELAPSED $SECONDS"
  done
done
