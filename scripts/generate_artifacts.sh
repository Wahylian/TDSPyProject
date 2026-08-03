#!/usr/bin/env bash
# Regenerate every committed run artifact under artifacts/<model>/<run_id>/.
#
# This is the complete, authoritative list of training commands: the 16 baseline
# exploratory runs plus the 53 full-matrix cells (69 valid model x pipeline
# pairings in total). Prepare the dataset first:
#
#     python src/download_dataset.py
#     python src/create_split.py
#
# Run the whole batch on Linux/macOS/CI with:  bash scripts/generate_artifacts.sh
# On Windows, drive it through PowerShell (see the README) since bash resolves to
# WSL2 and cannot see the Windows .venv.
#
# Conventions: --scoring f1 is the default (stated explicitly on the matrix cells
# so no cell can silently optimize a different metric); --diagnostics is limited
# to the mlp rows (the only estimator whose extra cost buys a learning curve the
# cheap diagnostics don't already give). All other defaults (--max-train-samples
# 5000, RANDOM_STATE=42, --cache-dir feature_cache) are left untouched so every
# cell shares one train/val/test budget, seed, and amortized feature cache.
set -e

# --- Baseline exploratory runs (16) --------------------------------------------

# Classical baselines on the default svm (PCA-150) front-end
python src/train_model.py --model svm    --pipeline svm
python src/train_model.py --model logreg --pipeline svm
python src/train_model.py --model ridge  --pipeline svm
python src/train_model.py --model linreg --pipeline svm
python src/train_model.py --model rf     --pipeline svm
python src/train_model.py --model hgb    --pipeline svm
python src/train_model.py --model mlp    --pipeline svm --diagnostics

# Hard-margin SVM pair (like-for-like)
python src/train_model.py --model hard_svm        --pipeline svm
python src/train_model.py --model hard_svm_kernel --pipeline svm

# Reduction-method resilience (same model, PCA vs JL vs learned embedding;
# svm x svm is the classical baseline above)
python src/train_model.py --model svm --pipeline svm_jl
python src/train_model.py --model svm --pipeline embedding_pca
python src/train_model.py --model svm --pipeline embedding_jl

# Torch image models on raw square pixels
python src/train_model.py --model cnn      --pipeline pixels
python src/train_model.py --model vit      --pipeline pixels
python src/train_model.py --model cnn_deep --pipeline pixels_hq
python src/train_model.py --model vit_deep --pipeline pixels_hq

# --- Full-matrix completion (53) -----------------------------------------------

# svm_jl column (JL reduce) — svm x svm_jl already in the baseline set
python src/train_model.py --model hard_svm        --pipeline svm_jl        --scoring f1
python src/train_model.py --model hard_svm_kernel --pipeline svm_jl        --scoring f1
python src/train_model.py --model logreg          --pipeline svm_jl        --scoring f1
python src/train_model.py --model ridge           --pipeline svm_jl        --scoring f1
python src/train_model.py --model linreg          --pipeline svm_jl        --scoring f1
python src/train_model.py --model rf              --pipeline svm_jl        --scoring f1
python src/train_model.py --model hgb             --pipeline svm_jl        --scoring f1
python src/train_model.py --model mlp             --pipeline svm_jl        --scoring f1 --diagnostics

# fast column (64x64, PCA-150) — full 9
python src/train_model.py --model svm             --pipeline fast          --scoring f1
python src/train_model.py --model hard_svm        --pipeline fast          --scoring f1
python src/train_model.py --model hard_svm_kernel --pipeline fast          --scoring f1
python src/train_model.py --model logreg          --pipeline fast          --scoring f1
python src/train_model.py --model ridge           --pipeline fast          --scoring f1
python src/train_model.py --model linreg          --pipeline fast          --scoring f1
python src/train_model.py --model rf              --pipeline fast          --scoring f1
python src/train_model.py --model hgb             --pipeline fast          --scoring f1
python src/train_model.py --model mlp             --pipeline fast          --scoring f1 --diagnostics

# hq column (224x224, PCA-300) — full 9
python src/train_model.py --model svm             --pipeline hq            --scoring f1
python src/train_model.py --model hard_svm        --pipeline hq            --scoring f1
python src/train_model.py --model hard_svm_kernel --pipeline hq            --scoring f1
python src/train_model.py --model logreg          --pipeline hq            --scoring f1
python src/train_model.py --model ridge           --pipeline hq            --scoring f1
python src/train_model.py --model linreg          --pipeline hq            --scoring f1
python src/train_model.py --model rf              --pipeline hq            --scoring f1
python src/train_model.py --model hgb             --pipeline hq            --scoring f1
python src/train_model.py --model mlp             --pipeline hq            --scoring f1 --diagnostics

# no_denoise column (svm minus denoise) — full 9
python src/train_model.py --model svm             --pipeline no_denoise    --scoring f1
python src/train_model.py --model hard_svm        --pipeline no_denoise    --scoring f1
python src/train_model.py --model hard_svm_kernel --pipeline no_denoise    --scoring f1
python src/train_model.py --model logreg          --pipeline no_denoise    --scoring f1
python src/train_model.py --model ridge           --pipeline no_denoise    --scoring f1
python src/train_model.py --model linreg          --pipeline no_denoise    --scoring f1
python src/train_model.py --model rf              --pipeline no_denoise    --scoring f1
python src/train_model.py --model hgb             --pipeline no_denoise    --scoring f1
python src/train_model.py --model mlp             --pipeline no_denoise    --scoring f1 --diagnostics

# embedding_pca column (VGG16 + PCA) — svm x embedding_pca already in the baseline set
python src/train_model.py --model hard_svm        --pipeline embedding_pca --scoring f1
python src/train_model.py --model hard_svm_kernel --pipeline embedding_pca --scoring f1
python src/train_model.py --model logreg          --pipeline embedding_pca --scoring f1
python src/train_model.py --model ridge           --pipeline embedding_pca --scoring f1
python src/train_model.py --model linreg          --pipeline embedding_pca --scoring f1
python src/train_model.py --model rf              --pipeline embedding_pca --scoring f1
python src/train_model.py --model hgb             --pipeline embedding_pca --scoring f1
python src/train_model.py --model mlp             --pipeline embedding_pca --scoring f1 --diagnostics

# embedding_jl column (VGG16 + JL) — svm x embedding_jl already in the baseline set
python src/train_model.py --model hard_svm        --pipeline embedding_jl  --scoring f1
python src/train_model.py --model hard_svm_kernel --pipeline embedding_jl  --scoring f1
python src/train_model.py --model logreg          --pipeline embedding_jl  --scoring f1
python src/train_model.py --model ridge           --pipeline embedding_jl  --scoring f1
python src/train_model.py --model linreg          --pipeline embedding_jl  --scoring f1
python src/train_model.py --model rf              --pipeline embedding_jl  --scoring f1
python src/train_model.py --model hgb             --pipeline embedding_jl  --scoring f1
python src/train_model.py --model mlp             --pipeline embedding_jl  --scoring f1 --diagnostics

# Pretrained torch backbones on 224x224 RGB pixels (the most expensive runs)
python src/train_model.py --model cnn_pretrained  --pipeline pixels_pretrained --scoring f1
python src/train_model.py --model vit_pretrained  --pipeline pixels_pretrained --scoring f1
