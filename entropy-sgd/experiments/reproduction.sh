python train.py \
        -B 400 \
        -exp-tag langevin_0_400_repr_s42 \
        -wandb-mode online \
        -b 100 \
        -lr 0.1 \
        -weight-decay 0.001 \
        -L 0 \
        -g 0 \
        -s 42 \
        -batch-step 100 \
        -lr-step 60 \
        -lr-decay 0.2 \
        -nesterov \
        -momentum 0.9 \
        -deterministic

python train.py \
        -B 20 \
        -exp-tag langevin_20_20_repr_s42 \
        -wandb-mode online \
        -b 100 \
        -lr 1 \
        -weight-decay 0.001 \
        -L 20 \
        -g 0 \
        -s 42 \
        -batch-step 100 \
        -lr-step 4 \
        -lr-decay 0.2 \
        -gamma 0.03 \
        -scoping 0.001 \
        -noise 0.0001 \
        -nesterov \
        -momentum 0.9 \
        -deterministic

python train.py \
        -B 400 \
        -exp-tag langevin_0_400_repr_s41 \
        -wandb-mode online \
        -b 100 \
        -lr 0.1 \
        -weight-decay 0.001 \
        -L 0 \
        -g 0 \
        -s 41 \
        -batch-step 100 \
        -lr-step 60 \
        -lr-decay 0.2 \
        -nesterov \
        -momentum 0.9 \
        -deterministic

python train.py \
        -B 20 \
        -exp-tag langevin_20_20_repr_s41 \
        -wandb-mode online \
        -b 100 \
        -lr 1 \
        -weight-decay 0.001 \
        -L 20 \
        -g 0 \
        -s 41 \
        -batch-step 100 \
        -lr-step 4 \
        -lr-decay 0.2 \
        -gamma 0.03 \
        -scoping 0.001 \
        -noise 0.0001 \
        -nesterov \
        -momentum 0.9 \
        -deterministic

python train.py \
        -B 400 \
        -exp-tag langevin_0_400_repr_s40 \
        -wandb-mode online \
        -b 100 \
        -lr 0.1 \
        -weight-decay 0.001 \
        -L 0 \
        -g 0 \
        -s 40 \
        -batch-step 100 \
        -lr-step 60 \
        -lr-decay 0.2 \
        -nesterov \
        -momentum 0.9 \
        -deterministic

python train.py \
        -B 20 \
        -exp-tag langevin_20_20_repr_s40 \
        -wandb-mode online \
        -b 100 \
        -lr 1 \
        -weight-decay 0.001 \
        -L 20 \
        -g 0 \
        -s 40 \
        -batch-step 100 \
        -lr-step 4 \
        -lr-decay 0.2 \
        -gamma 0.03 \
        -scoping 0.001 \
        -noise 0.0001 \
        -nesterov \
        -momentum 0.9 \
        -deterministic
