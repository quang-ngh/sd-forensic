SD_CKPT=editing/stable-diffusion-v1-5
EXPNAME=$1
LOSS="l1"
STEPS=100
python loss_region.py \
        --checkpoint-path ${SD_CKPT} \
        --experiment-name ${EXPNAME} \
        --inpaint-strength 1.0 \
        --start 0 \
        --end ${STEPS} \
        --inference-steps ${STEPS} \
        --guidance-scale 1.0 \
        --loss-type ${LOSS} \
        --real-image False