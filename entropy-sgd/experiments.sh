# # # We train for 200 epochs with SGD and Nesterov’s momentum during 
# # # which the initial learning rate of 0.1 decreases by a factor of 5 after every 60 epochs

# # # We train Entropy-SGD with L = 20 for 10 epochs with the original dropout of 0.5. The initial
# # # learning rate of the outer loop is set to η = 1 and drops by a factor of 5 every 4 epochs, while the
# # # learning rate of the SGLD updates is fixed to η0 = 0.1 with thermal noise ε = 10−4
# # # As the scoping scheme, we set the initial value of the scope to γ0 = 0.03 which increases 
# # # by a factor of 1.001 after each parameter update.

# echo -e "Entropy SGD, L=20, seed 51, no scoping, gamma=3"

# python train.py \
#         -B 10 \
#         -exp-tag entropy_allcnn_L20_s51_g3_lrdecay \
#         -wandb-mode online \
#         -m allcnn \
#         -b 128 \
#         -eval-b 1000 \
#         --lr 1 \
#         --l2 0 \
#         -L 20 \
#         -g 0 \
#         -s 51 \
#         -epoch-step 2 \
#         -batch-step 2 \
#         --lr-step 4 \
#         --lr-decay 0.2 \
#         --gamma 3 \
#         --noise 0.0001 \
#         --nesterov

# echo -e "Entropy SGD, L=20, seed 51, no scoping, gamma=0.3"

# python train.py \
#         -B 10 \
#         -exp-tag entropy_allcnn_L20_s51_gp3_lrdecay \
#         -wandb-mode online \
#         -m allcnn \
#         -b 128 \
#         -eval-b 1000 \
#         --lr 1 \
#         --l2 0 \
#         -L 20 \
#         -g 0 \
#         -s 51 \
#         -epoch-step 1 \
#         -batch-step 2 \
#         --lr-step 4 \
#         --lr-decay 0.2 \
#         --gamma 0.3 \
#         --noise 0.0001 \
#         --nesterov

# echo -e "Entropy SGD, L=20, seed 51, no scoping, gamma=0.03"

# python train.py \
#         -B 10 \
#         -exp-tag entropy_allcnn_L20_s51_gp03_lrdecay \
#         -wandb-mode online \
#         -m allcnn \
#         -b 128 \
#         -eval-b 1000 \
#         --lr 1 \
#         --l2 0 \
#         -L 20 \
#         -g 0 \
#         -s 51 \
#         -epoch-step 2 \
#         -batch-step 2 \
#         --lr-step 4 \
#         --lr-decay 0.2 \
#         --gamma 0.03 \
#         --noise 0.0001 \
#         --nesterov

# echo -e "Entropy SGD, L=20, seed 51, no scoping, gamma=0.03"

# python train.py \
#         -B 10 \
#         -exp-tag entropy_allcnn_L20_s51_gsp03_lrdecay \
#         -wandb-mode online \
#         -m allcnn \
#         -b 128 \
#         -eval-b 1000 \
#         --lr 1 \
#         --l2 0 \
#         -L 20 \
#         -g 0 \
#         -s 51 \
#         -epoch-step 2 \
#         -batch-step 2 \
#         --lr-step 4 \
#         --lr-decay 0.2 \
#         --gamma 0.03 \
#         --noise 0.0001 \
#         --nesterov \
#         --apply-scoping

# echo -e "Entropy SGD, L=20, seed 51, no scoping, gamma=0.003"

# python train.py \
#         -B 10 \
#         -exp-tag entropy_allcnn_L20_s51_gp003_lrdecay \
#         -wandb-mode online \
#         -m allcnn \
#         -b 128 \
#         -eval-b 1000 \
#         --lr 1 \
#         --l2 0 \
#         -L 20 \
#         -g 0 \
#         -s 51 \
#         -epoch-step 2 \
#         -batch-step 2 \
#         --lr-step 4 \
#         --lr-decay 0.2 \
#         --gamma 0.003 \
#         --noise 0.0001 \
#         --nesterov

# echo -e "Entropy SGD, L=20, seed 51, gs=0.003"

# python train.py \
#         -B 10 \
#         -exp-tag entropy_allcnn_L20_s51_gsp003_lrdecay \
#         -wandb-mode online \
#         -m allcnn \
#         -b 128 \
#         -eval-b 1000 \
#         --lr 1 \
#         --l2 0 \
#         -L 20 \
#         -g 0 \
#         -s 51 \
#         -epoch-step 2 \
#         -batch-step 2 \
#         --lr-step 4 \
#         --lr-decay 0.2 \
#         --gamma 0.003 \
#         --noise 0.0001 \
#         --nesterov \
#         --apply-scoping

# echo -e "Entropy SGD, L=20, seed 51, g=0.0003"

# python train.py \
#         -B 10 \
#         -exp-tag entropy_allcnn_L20_s51_gp0003_lrdecay \
#         -wandb-mode online \
#         -m allcnn \
#         -b 128 \
#         -eval-b 1000 \
#         --lr 1 \
#         --l2 0 \
#         -L 20 \
#         -g 0 \
#         -s 51 \
#         -epoch-step 2 \
#         -batch-step 2 \
#         --lr-step 4 \
#         --lr-decay 0.2 \
#         --gamma 0.0003 \
#         --noise 0.0001 \
#         --nesterov

# echo -e "Entropy SGD, L=20, seed 51, gs=0.0003"

# python train.py \
#         -B 10 \
#         -exp-tag entropy_allcnn_L20_s51_gsp0003_lrdecay \
#         -wandb-mode online \
#         -m allcnn \
#         -b 128 \
#         -eval-b 1000 \
#         --lr 1 \
#         --l2 0 \
#         -L 20 \
#         -g 0 \
#         -s 51 \
#         -epoch-step 2 \
#         -batch-step 2 \
#         --lr-step 4 \
#         --lr-decay 0.2 \
#         --gamma 0.0003 \
#         --noise 0.0001 \
#         --nesterov \
#         --apply-scoping

# echo -e "Entropy SGD, L=20, seed 51, g=0.00003"

# python train.py \
#         -B 10 \
#         -exp-tag entropy_allcnn_L20_s51_gp00003_lrdecay \
#         -wandb-mode online \
#         -m allcnn \
#         -b 128 \
#         -eval-b 1000 \
#         --lr 1 \
#         --l2 0 \
#         -L 20 \
#         -g 0 \
#         -s 51 \
#         -epoch-step 2 \
#         -batch-step 2 \
#         --lr-step 4 \
#         --lr-decay 0.2 \
#         --gamma 0.00003 \
#         --noise 0.0001 \
#         --nesterov

echo -e "Entropy SGD, L=20, seed 51, gs=0.00003"

python train.py \
        -B 10 \
        -exp-tag entropy_allcnn_L20_s51_gsp00003_lrdecay \
        -wandb-mode online \
        -m allcnn \
        -b 128 \
        -eval-b 1000 \
        --lr 1 \
        --l2 0 \
        -L 20 \
        -g 0 \
        -s 51 \
        -epoch-step 2 \
        -batch-step 2 \
        --lr-step 4 \
        --lr-decay 0.2 \
        --gamma 0.00003 \
        --noise 0.0001 \
        --nesterov \
        --apply-scoping


echo -e "ENTROPY SGD, L=25, seed 51, no scoping"
    
python train.py \
        -B 8 \
        -exp-tag entropy_all_cnn_L_25_default_seed51_lrdecay \
        -wandb-mode online \
        -m allcnn \
        -b 128 \
        -eval-b 1000 \
        --lr 1 \
        --l2 0 \
        -L 25 \
        -g 0 \
        -s 51 \
        -epoch-step 2 \
        -batch-step 2 \
        --lr-step 3 \
        --lr-decay 0.2 \
        --gamma 0.03 \
        --noise 0.0001 \
        --nesterov

echo -e "ENTROPY SGD, L=10, seed 51, no scoping"
    
python train.py \
        -B 20 \
        -exp-tag entropy_all_cnn_L_10_default_seed51_lrdecay \
        -wandb-mode online \
        -m allcnn \
        -b 128 \
        -eval-b 1000 \
        --lr 1 \
        --l2 0 \
        -L 10 \
        -g 0 \
        -s 51 \
        -epoch-step 4 \
        -batch-step 4 \
        --lr-step 8 \
        --lr-decay 0.2 \
        --gamma 0.03 \
        --noise 0.0001 \
        --nesterov

echo -e "ENTROPY SGD, L=5, seed 51,no scoping"
    
python train.py \
        -B 40 \
        -exp-tag entropy_all_cnn_L_5_default_seed51_lrdecay \
        -wandb-mode online \
        -m allcnn \
        -b 128 \
        -eval-b 1000 \
        --lr 1 \
        --l2 0 \
        -L 5 \
        -g 0 \
        -s 51 \
        -epoch-step 8 \
        -batch-step 8 \
        --lr-step 16 \
        --lr-decay 0.2 \
        --gamma 0.03 \
        --noise 0.0001 \
        --nesterov

echo -e "ENTROPY SGD, L=20, seed 50, with scoping"

python train.py \
        -B 10 \
        -exp-tag entropy_all_cnn_L_20_default_seed50_lrdecay \
        -wandb-mode online \
        -m allcnn \
        -b 128 \
        -eval-b 1000 \
        --lr 1 \
        --l2 0 \
        -L 20 \
        -g 0 \
        -s 50 \
        -epoch-step 2 \
        -batch-step 2 \
        --lr-step 4 \
        --lr-decay 0.2 \
        --gamma 0.03 \
        --noise 0.0001 \
        --nesterov \
        --apply-scoping

echo -e "ENTROPY SGD, L=20, seed 52, with scoping"

python train.py \
        -B 10 \
        -exp-tag entropy_all_cnn_L_20_default_seed52_lrdecay \
        -wandb-mode online \
        -m allcnn \
        -b 128 \
        -eval-b 1000 \
        --lr 1 \
        --l2 0 \
        -L 20 \
        -g 0 \
        -s 52 \
        -epoch-step 2 \
        -batch-step 2 \
        --lr-step 4 \
        --lr-decay 0.2 \
        --gamma 0.03 \
        --noise 0.0001 \
        --nesterov \
        --apply-scoping

echo -e "SGD seed 50"

python train.py \
        -B 200 \
        -exp-tag sgd_all_cnn_seed50_lrdecay \
        -wandb-mode online \
        -m allcnn \
        -b 128 \
        -eval-b 1000 \
        --lr 0.1 \
        --l2 0 \
        -L 0 \
        -g 0 \
        -s 50 \
        -epoch-step 40 \
        -batch-step 40 \
        --lr-step 60 \
        --lr-decay 0.2

echo -e "SGD seed 51"

python train.py \
        -B 200 \
        -exp-tag sgd_all_cnn_seed51_lrdecay \
        -wandb-mode online \
        -m allcnn \
        -b 100 \
        -eval-b 1000 \
        --lr 0.1 \
        --l2 0 \
        -L 0 \
        -g 0 \
        -s 51 \
        -epoch-step 100 \
        -batch-step 100 \
        --lr-step 60 \
        --lr-decay 0.2

echo -e "SGD seed 52"

python train.py \
        -B 200 \
        -exp-tag sgd_all_cnn_seed52_lrdecay \
        -wandb-mode online \
        -m allcnn \
        -b 128 \
        -eval-b 1000 \
        --lr 0.1 \
        --l2 0 \
        -L 0 \
        -g 0 \
        -s 52 \
        -epoch-step 40 \
        -batch-step 40 \
        --lr-step 60 \
        --lr-decay 0.2