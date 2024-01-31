#   Data path
DATASET=datasets/AutoSplice
# PRED_DIR=experiments/reproduce_baselines/pscc_inference_forged100/predictions
PRED_DIR=experiments/reproduce_baselines/pscc_inference_forged100_2
SAVE_NAME=AutoSplice_2

CUDA=0

CUDA_VISIBLE_DEVICES=${CUDA} nohup python inference.py >> logs/pscc_magicbrush.txt &

# python tools/eval_baselines.py --gt-dir ${DATASET}/Mask --pred-dir ${PRED_DIR} --save-name ${SAVE_NAME}
