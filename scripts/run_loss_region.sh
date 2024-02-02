SD_CKPT=../checkpoints/sd_v15
EXPNAME=$1
LOSS="l2"
STEPS=50
CUDA=0

CUDA_VISIBLE_DEVICES=${CUDA} python loss_region.py \
                        --checkpoint-path ${SD_CKPT} \
                        --experiment-name ${EXPNAME} \
                        --inpaint-strength 1.0 \
                        --start 0 \
                        --end ${STEPS} \
                        --inference-steps ${STEPS} \
                        --guidance-scale 7.5 \
                        --loss-type ${LOSS} \
                        --concat-images