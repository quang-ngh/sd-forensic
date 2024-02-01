SD_CKPT=editing/stable-diffusion-v1-5
EXPNAME=$1
LOSS="l2"
STEPS=50
python loss_region.py \
        --checkpoint-path ${SD_CKPT} \
        --experiment-name ${EXPNAME} \
        --inpaint-strength 1.0 \
        --start 0 \
        --end ${STEPS} \
        --inference-steps ${STEPS} \
        --guidance-scale 7.5 \
        --loss-type ${LOSS} \
        # --real-image False