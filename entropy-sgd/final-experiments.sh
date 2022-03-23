# # # We train for 200 epochs with SGD and Nesterov’s momentum during 
# # # which the initial learning rate of 0.1 decreases by a factor of 5 after every 60 epochs

# # # We train Entropy-SGD with L = 20 for 10 epochs with the original dropout of 0.5. The initial
# # # learning rate of the outer loop is set to η = 1 and drops by a factor of 5 every 4 epochs, while the
# # # learning rate of the SGLD updates is fixed to η0 = 0.1 with thermal noise ε = 10−4
# # # As the scoping scheme, we set the initial value of the scope to γ0 = 0.03 which increases 
# # # by a factor of 1.001 after each parameter update.

# Date: 23/03/2022
# Reproduction experiments trying the stopping criteria
python train.py \
        -B 10 \
        -exp-tag entropy_reproduction \
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
        -apply-scoping \
        -min-loss 0.1 \
        -calculate
    
python train.py \
        -B 200 \
        -exp-tag sgd_reproduction \
        -wandb-mode online \
        -m allcnn \
        -b 100 \
        -eval-b 100 \
        -lr 0.1 \
        -l2 0 \
        -L 0 \
        -g 0 \
        -s 51 \
        -epoch-step 20 \
        -batch-step 100 \
        -lr-step 60 \
        -lr-decay 0.2 \
        -nesterov \
        -momentum 0.9 \
        -min-loss 0.1 \
        -calculate
