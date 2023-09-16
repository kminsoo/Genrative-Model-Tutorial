#!/usr/bin/env bash
mi_distill=$1
step_size=$2
noise_sigma_t=$3
l2_coeff=0.05
langevin_steps=$4
gpu_ids=$5
log_dir="logs/unet_pix2pix/edges2shoes-r/VEM_mi_distill_${mi_distill}_step_size_${step_size}_l2_coeff_${l2_coeff}_fixed_${noise_sigma_t}_noise_langevin_steps_${langevin_steps}" 
python distill.py --dataroot database/edges2shoes-r \
  --gpu_ids ${gpu_ids} \
  --print_freq 100 --n_share 5 \
  --lambda_CD 1e1 \
  --distiller multiteacher_mi \
  --log_dir ${log_dir} \
  --batch_size 4 --num_teacher 2 \
  --real_stat_path real_stat/edges2shoes-r_B.npz \
  --teacher_ngf_w 64 --teacher_ngf_d 16 --student_ngf 16  --norm batch \
  --teacher_netG_w unet_256 --teacher_netG_d unet_deepest_256 --netD multi_n_layers \
  --nepochs 100 --nepochs_decay 100 --n_dis 1 \
  --AGD_weights 1e1,1e4,1e1,1e-5 \
  --lambda_mi_distill ${mi_distill} \
  --lambda_step_size ${step_size} \
  --lambda_sigma_t ${noise_sigma_t} \
  --lambda_l2_coeff ${l2_coeff} \
  --langevin_steps ${langevin_steps} \
