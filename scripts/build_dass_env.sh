#!/bin/bash
# Build `.venv-dass`: the training environment plus the DASS selective-scan CUDA kernel, compiled with CUDA 12.8 so that the
# system CUDA (any version) is left alone. Needs miniconda and ../repos/DASS (scripts/download_data.sh clones it).
#   bash scripts/build_dass_env.sh
# Why a conda toolkit: the kernel must be compiled with the same CUDA major.minor as torch (cu128); conda ships nvcc 12.8 and its
# own g++, but its g++ does not search /usr/include, hence the small private include folder for the Python architecture headers.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH=$HOME/.local/bin:$PATH
CONDA_ENV=cuda128
source ~/miniconda3/etc/profile.d/conda.sh
conda env list | grep -q "^$CONDA_ENV " || conda create -y -n $CONDA_ENV -c nvidia/label/cuda-12.8.1 -c conda-forge cuda-toolkit=12.8 gxx_linux-64=11
C=$(conda info --base)/envs/$CONDA_ENV
IDX="--index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match"
[ -d .venv-dass ] || uv venv .venv-dass --python 3.10
uv pip install --python .venv-dass/bin/python $IDX -r requirements-train.lock ninja packaging einops fvcore setuptools wheel

PYINC=${TMPDIR:-/tmp}/pyinc-$USER
mkdir -p $PYINC/x86_64-linux-gnu && ln -sfn /usr/include/x86_64-linux-gnu/python3.10 $PYINC/x86_64-linux-gnu/python3.10
export CUDA_HOME=$C PATH=$C/bin:$PATH CC=$C/bin/x86_64-conda-linux-gnu-gcc CXX=$C/bin/x86_64-conda-linux-gnu-g++ TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"  # only used by torch, not by this setup.py
export CPATH=$PYINC:$C/targets/x86_64-linux/include:$C/include LIBRARY_PATH=$C/lib:$C/targets/x86_64-linux/lib
export LD_LIBRARY_PATH=$C/lib:$C/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
# the kernel's setup.py hardcodes sm_70/80/90; build from a copy with the GPU's own architecture added (the repo stays untouched)
SM=$(.venv-dass/bin/python -c "import torch;m,n=torch.cuda.get_device_capability(0);print(f'{m}{n}')")
BUILD=${TMPDIR:-/tmp}/selective_scan-$USER; rm -rf $BUILD; cp -r ../repos/DASS/kernels/selective_scan $BUILD
sed -i "65a\\    cc_flag.extend(['-gencode', 'arch=compute_$SM,code=sm_$SM'])" $BUILD/setup.py
uv pip install --python "$PWD/.venv-dass/bin/python" --no-build-isolation --reinstall $BUILD
echo "kernel built; at run time export LD_LIBRARY_PATH=$C/lib (scripts/run.sh does this for PY=.venv-dass/bin/python)"
