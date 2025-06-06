<div align="center">

# Multi-stage Bayesian Prototype Refinement with Feature Weighting for Few-shot Classification

</div>

<!-- ## Setting Up -->

### Package Dependencies
1. Create a new Conda environment with Python 3.7 then activate it:
```shell
conda create -n MBPRFW python==3.7
conda activate MBPRFW
```

2. Install PyTorch v1.12.1 with a CUDA version that works on your cluster/machine (CUDA 11.3 is used in this example):
```shell
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```

3. Install the packages in `requirements.txt` via `pip`:
```shell
pip install -r requirements.txt
```