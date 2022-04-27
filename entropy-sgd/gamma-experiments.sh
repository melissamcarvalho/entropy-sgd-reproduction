python train.py \
        -B 10 \
        -exp-tag gamma_p5_scoping \
        -wandb-mode online \
        -m allcnn \
        -b 100 \
        -eval-b 100 \
        -lr 1 \
        -l2 0 \
        -L 20 \
        -g 0 \
        -s 51 \
        -epoch-step 1 \
        -batch-step 100 \
        -lr-step 4 \
        -lr-decay 0.2 \
        -gamma 0.00003 \
        -scoping 0.001 \
        -noise 0.0001 \
        -nesterov \
        -momentum 0.9 \
        -apply-scoping \
        -min-loss 0.1 \
        -calculate

python train.py \
        -B 10 \
        -exp-tag gamma_p4_scoping \
        -wandb-mode online \
        -m allcnn \
        -b 100 \
        -eval-b 100 \
        -lr 1 \
        -l2 0 \
        -L 20 \
        -g 0 \
        -s 51 \
        -epoch-step 1 \
        -batch-step 100 \
        -lr-step 4 \
        -lr-decay 0.2 \
        -gamma 0.0003 \
        -scoping 0.001 \
        -noise 0.0001 \
        -nesterov \
        -momentum 0.9 \
        -apply-scoping \
        -min-loss 0.1 \
        -calculate

python train.py \
        -B 10 \
        -exp-tag gamma_p3_scoping \
        -wandb-mode online \
        -m allcnn \
        -b 100 \
        -eval-b 100 \
        -lr 1 \
        -l2 0 \
        -L 20 \
        -g 0 \
        -s 51 \
        -epoch-step 1 \
        -batch-step 100 \
        -lr-step 4 \
        -lr-decay 0.2 \
        -gamma 0.003 \
        -scoping 0.001 \
        -noise 0.0001 \
        -nesterov \
        -momentum 0.9 \
        -apply-scoping \
        -min-loss 0.1 \
        -calculate

python train.py \
        -B 10 \
        -exp-tag gamma_p1_scoping \
        -wandb-mode online \
        -m allcnn \
        -b 100 \
        -eval-b 100 \
        -lr 1 \
        -l2 0 \
        -L 20 \
        -g 0 \
        -s 51 \
        -epoch-step 1 \
        -batch-step 100 \
        -lr-step 4 \
        -lr-decay 0.2 \
        -gamma 0.3 \
        -scoping 0.001 \
        -noise 0.0001 \
        -nesterov \
        -momentum 0.9 \
        -apply-scoping \
        -min-loss 0.1 \
        -calculate

python train.py \
        -B 10 \
        -exp-tag gamma_3_scoping \
        -wandb-mode online \
        -m allcnn \
        -b 100 \
        -eval-b 100 \
        -lr 1 \
        -l2 0 \
        -L 20 \
        -g 0 \
        -s 51 \
        -epoch-step 1 \
        -batch-step 100 \
        -lr-step 4 \
        -lr-decay 0.2 \
        -gamma 3 \
        -scoping 0.001 \
        -noise 0.0001 \
        -nesterov \
        -momentum 0.9 \
        -apply-scoping \
        -min-loss 0.1 \
        -calculate

python train.py \
        -B 10 \
        -exp-tag gamma_p5_noscoping \
        -wandb-mode online \
        -m allcnn \
        -b 100 \
        -eval-b 100 \
        -lr 1 \
        -l2 0 \
        -L 20 \
        -g 0 \
        -s 51 \
        -epoch-step 1 \
        -batch-step 100 \
        -lr-step 4 \
        -lr-decay 0.2 \
        -gamma 0.00003 \
        -scoping 0.001 \
        -noise 0.0001 \
        -nesterov \
        -momentum 0.9 \
        -min-loss 0.1 \
        -calculate

python train.py \
        -B 10 \
        -exp-tag gamma_p4_noscoping \
        -wandb-mode online \
        -m allcnn \
        -b 100 \
        -eval-b 100 \
        -lr 1 \
        -l2 0 \
        -L 20 \
        -g 0 \
        -s 51 \
        -epoch-step 1 \
        -batch-step 100 \
        -lr-step 4 \
        -lr-decay 0.2 \
        -gamma 0.0003 \
        -scoping 0.001 \
        -noise 0.0001 \
        -nesterov \
        -momentum 0.9 \
        -min-loss 0.1 \
        -calculate

python train.py \
        -B 10 \
        -exp-tag gamma_p3_noscoping \
        -wandb-mode online \
        -m allcnn \
        -b 100 \
        -eval-b 100 \
        -lr 1 \
        -l2 0 \
        -L 20 \
        -g 0 \
        -s 51 \
        -epoch-step 1 \
        -batch-step 100 \
        -lr-step 4 \
        -lr-decay 0.2 \
        -gamma 0.003 \
        -scoping 0.001 \
        -noise 0.0001 \
        -nesterov \
        -momentum 0.9 \
        -min-loss 0.1 \
        -calculate

python train.py \
        -B 10 \
        -exp-tag gamma_p2_noscoping \
        -wandb-mode online \
        -m allcnn \
        -b 100 \
        -eval-b 100 \
        -lr 1 \
        -l2 0 \
        -L 20 \
        -g 0 \
        -s 51 \
        -epoch-step 1 \
        -batch-step 100 \
        -lr-step 4 \
        -lr-decay 0.2 \
        -gamma 0.03 \
        -scoping 0.001 \
        -noise 0.0001 \
        -nesterov \
        -momentum 0.9 \
        -min-loss 0.1 \
        -calculate

python train.py \
        -B 10 \
        -exp-tag gamma_p1_noscoping \
        -wandb-mode online \
        -m allcnn \
        -b 100 \
        -eval-b 100 \
        -lr 1 \
        -l2 0 \
        -L 20 \
        -g 0 \
        -s 51 \
        -epoch-step 1 \
        -batch-step 100 \
        -lr-step 4 \
        -lr-decay 0.2 \
        -gamma 0.3 \
        -scoping 0.001 \
        -noise 0.0001 \
        -nesterov \
        -momentum 0.9 \
        -min-loss 0.1 \
        -calculate

python train.py \
        -B 10 \
        -exp-tag gamma_3_noscoping \
        -wandb-mode online \
        -m allcnn \
        -b 100 \
        -eval-b 100 \
        -lr 1 \
        -l2 0 \
        -L 20 \
        -g 0 \
        -s 51 \
        -epoch-step 1 \
        -batch-step 100 \
        -lr-step 4 \
        -lr-decay 0.2 \
        -gamma 3 \
        -scoping 0.001 \
        -noise 0.0001 \
        -nesterov \
        -momentum 0.9 \
        -min-loss 0.1 \
        -calculate
