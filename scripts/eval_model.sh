DEVICES="0"
NUM_WORKERS=0

#EVAL_DATASET="data/tsp/tsp20_test_concorde.txt" "data/tsp/tsp50_test_concorde.txt" "data/tsp/tsp100_test_concorde.txt" "data/tsp/tsp200_test_concorde.txt"
VAL_SIZE=25600

BATCH_SIZE=32

CUDA_VISIBLE_DEVICES="$DEVICES" python eval.py  \
        "data/tsp/tsp20_test_concorde.txt" "data/tsp/tsp50_test_concorde.txt" "data/tsp/tsp100_test_concorde.txt" \
        --val_size "$VAL_SIZE" --batch_size "$BATCH_SIZE" \
        --model "$1" \
        --decode_strategies "greedy"\
        --widths 0 \
        --num_workers "$NUM_WORKERS"