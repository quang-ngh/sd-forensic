BASELINE="pscc"
DATASET=AutoSplice

GT=datasets/${DATASET}/Mask
PRED_DIR=experiments/reproduce_baselines/pscc_inference_forged100_epoch100

SAVENAME=AutoSplice_forged_epoch100

python tools/eval_baselines.py --gt-dir ${GT} --pred-dir ${PRED_DIR} \
        --dataset ${DATASET} --save-name ${SAVENAME} --baseline-name ${BASELINE} \