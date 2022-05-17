python train.py \
        -B 10 \
        -exp-tag test_entropy \
        -wandb-mode online \
        -b 100 \
        -eval-b 100 \
        -lr 1 \
        -l2 0 \
        -L 20 \
        -g 0 \
        -s 42 \
        -epoch-step 1 \
        -batch-step 100 \
        -lr-step 4 \
        -lr-decay 0.2 \
        -gamma 0.03 \
        -scoping 0.001 \
        -noise 0.0001 \
        -nesterov \
        -momentum 0.9 \
        -apply-scoping \
        -deterministic \
        -dropout 0.5
    
python train.py \
        -B 200 \
        -exp-tag test_sgd \
        -wandb-mode online \
        -b 100 \
        -eval-b 100 \
        -lr 0.1 \
        -l2 0 \
        -L 0 \
        -g 0 \
        -s 42 \
        -epoch-step 20 \
        -batch-step 100 \
        -lr-step 60 \
        -lr-decay 0.2 \
        -gamma 0.03 \
        -scoping 0.001 \
        -noise 0.0001 \
        -nesterov \
        -momentum 0.9 \
        -apply-scoping \
        -deterministic
