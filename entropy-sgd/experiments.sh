echo -e "Initial test"

python train.py \
        -B 1 \
        -exp-tag initial_test_one_epoch_allcnn \
        -wandb-mode disabled \
        -m allcnn \
        -b 128 \
        -eval-b 1000 \
        --lr 0.1 \
        --l2 0 \
        -L 0 \
        --gamma 0 \
        --scoping 0 \
        --noise 0 \
        -g 0 \
        -s 51 \
        -epoch-step 100 \
        -batch-step 100
