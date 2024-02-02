DATASET=datasets/DiffEdit
MVSS_CKPT=MVSS-Net/ckpt/mvssnet_casia.pt
MVSSPLUS_CKPT=MVSS-Net/ckpt/mvssnetplus_casia.pt

#   CUDA device
CUDA=0

#   Inference to get the binary mask
EXP_NAME=mvss_inference_DiffEdithard_AutoSplice
SAVE_DIR=experiments/reproduce_baselines/${EXP_NAME}/predictions
TEST_DIR=${DATASET}/DiffEdit_hard
LOG_DIR=logs/

#   StdOut to text file
CUDA_VISIBLE_DEVICES=${CUDA} nohup python MVSS-Net/inference_new.py \
                                    --model-path ${MVSS_CKPT} \
                                    --test-dir ${TEST_DIR} \
                                    --save-dir ${SAVE_DIR} \
                                    --resize 512 \
                                    >> ${LOG_DIR}/${EXP_NAME}.txt &

#   Stdout to terminal
# python MVSS-Net/inference_new.py \
#                                     --model-path ${MVSS_CKPT} \
#                                     --test-dir ${TEST_DIR} \
#                                     --save-dir ${SAVE_DIR} \
#                                     --resize 512 
#   Evaluate to calculate the IoU and Accuracy + F1 score 
#   We need to analyse the performance of the models w.r.t the datasets

