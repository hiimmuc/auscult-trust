#!/bin/bash
# Patch-Mix CL reproduction: exact args of scripts/icbhi_patchmix_cl.sh, one seed per call.
s=$1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
.venv/bin/python main.py --tag bs8_lr5e-5_ep50_seed${s}_best_param --dataset icbhi --seed $s --class_split lungsound \
  --n_cls 4 --epochs 50 --batch_size 8 --optimizer adam --learning_rate 5e-5 --weight_decay 1e-6 --cosine \
  --model ast --test_fold official --pad_types repeat --resz 1 --n_mels 128 --ma_update --ma_beta 0.5 \
  --from_sl_official --audioset_pretrained --method patchmix_cl --temperature 0.06 --proj_dim 768 \
  --alpha 1.0 --mix_beta 1.0 --resume save/icbhi_ast_patchmix_cl_bs8_lr5e-5_ep50_seed${s}_best_param/last.pth
