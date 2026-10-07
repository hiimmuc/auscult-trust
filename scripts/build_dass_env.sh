#!/bin/bash
# Build `.venv-dass`: the training environment plus the DASS selective-scan CUDA kernel.
#   bash scripts/build_dass_env.sh            torch 2.9.1 built for CUDA 13.0 (default), kernel compiled with the system CUDA 13.0
#   CUDA_TAG=cu128 bash scripts/build_dass_env.sh   torch 2.9.1 for CUDA 12.8, kernel compiled with a conda CUDA 12.8 toolkit
# The kernel must be compiled with the same CUDA major version as torch (PyTorch's extension builder refuses otherwise).
# The kernel's setup.py hardcodes sm_70/80/90, so it is built from a copy with the GPU's own architecture added; ../repos/DASS
# stays untouched. The kernel is installed with --reinstall-package so that torch is never replaced by a newer PyPI build.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH=$HOME/.local/bin:$PATH
CUDA_TAG=${CUDA_TAG:-cu130}
PY=$PWD/.venv-dass/bin/python
[ -d .venv-dass ] || uv venv .venv-dass --python 3.10

# 1. torch trio from the PyTorch index of the chosen CUDA build, then the rest of the pinned training environment
uv pip install --python $PY torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/$CUDA_TAG --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match
grep -v -E '^(torch|torchvision|torchaudio|triton|nvidia-)' requirements-train.lock > ${TMPDIR:-/tmp}/lock-rest-$USER.txt
uv pip install --python $PY -r ${TMPDIR:-/tmp}/lock-rest-$USER.txt ninja packaging einops fvcore setuptools wheel

# 2. CUDA toolkit used to compile the kernel
if [ "$CUDA_TAG" = cu130 ]; then
  export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH
  export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
else
  source ~/miniconda3/etc/profile.d/conda.sh
  conda env list | grep -q "^cuda128 " || conda create -y -n cuda128 -c nvidia/label/cuda-12.8.1 -c conda-forge cuda-toolkit=12.8 gxx_linux-64=11
  C=$(conda info --base)/envs/cuda128
  PYINC=${TMPDIR:-/tmp}/pyinc-$USER  # conda's g++ does not search /usr/include: expose the Python architecture headers
  mkdir -p $PYINC/x86_64-linux-gnu && ln -sfn /usr/include/x86_64-linux-gnu/python3.10 $PYINC/x86_64-linux-gnu/python3.10
  export CUDA_HOME=$C PATH=$C/bin:$PATH CC=$C/bin/x86_64-conda-linux-gnu-gcc CXX=$C/bin/x86_64-conda-linux-gnu-g++
  export CPATH=$PYINC:$C/targets/x86_64-linux/include:$C/include LIBRARY_PATH=$C/lib:$C/targets/x86_64-linux/lib
  export LD_LIBRARY_PATH=$C/lib:$C/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
fi

# 3. kernel, built for the GPU in this machine
SM=$($PY -c "import torch;m,n=torch.cuda.get_device_capability(0);print(f'{m}{n}')")
export TORCH_CUDA_ARCH_LIST="${SM:0:-1}.${SM: -1}"
BUILD=${TMPDIR:-/tmp}/selective_scan-$USER; rm -rf $BUILD; cp -r ../repos/DASS/kernels/selective_scan $BUILD
sed -i "65a\\    cc_flag.extend(['-gencode', 'arch=compute_$SM,code=sm_$SM'])" $BUILD/setup.py
uv pip install --python $PY --no-build-isolation --reinstall-package selective-scan $BUILD
$PY -c "import torch,torchvision,selective_scan_cuda_oflex;print('torch',torch.__version__,'torchvision',torchvision.__version__,'kernel ok')"
