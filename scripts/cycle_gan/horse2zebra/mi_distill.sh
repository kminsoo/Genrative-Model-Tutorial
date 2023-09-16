#!/usr/bin/env bash
mi_distill=$1
step_size=$2
noise_sigma_t=$3
l2_coeff=0.05
langevin_steps=$4
gpu_ids=$5
log_dir="logs/cycle_gan/horse2zebra/VEM_mi_distill_${mi_distill}_step_size_${step_size}_l2_coeff_${l2_coeff}_fixed_${noise_sigma_t}_noise_langevin_steps_${langevin_steps}" 

python distill.py --dataroot database/horse2zebra \
  --gpu_ids ${gpu_ids} \
  --distiller cycleganbest_mi \
  --log_dir ${log_dir} \
  --real_stat_A_path real_stat/horse2zebra_A.npz \
  --real_stat_B_path real_stat/horse2zebra_B.npz \
  --teacher_ngf 64 --student_ngf 16 \
  --lambda_CD 5e2 \
  --teacher_netG mobile_resnet_9blocks --student_netG mobile_resnet_9blocks \
  --nepochs 100 --nepochs_decay 100 --n_dis 4 \
  --save_latest_freq 25000 --save_epoch_freq 10 \
  --AGD_weights 1e1,1e4,1e1,1e-5 \
  --lambda_mi_distill ${mi_distill} \
  --lambda_step_size ${step_size} \
  --lambda_sigma_t ${noise_sigma_t} \
  --lambda_l2_coeff ${l2_coeff} \
  --langevin_steps ${langevin_steps} \
 
