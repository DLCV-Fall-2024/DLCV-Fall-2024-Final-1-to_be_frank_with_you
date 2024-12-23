# Download Dataset

```bash
python -m utils.dataset
```
# Prerequisites

1. You need to install `natten` package first by running the command in [website](https://www.shi-labs.com/natten/), e.g.:
```bash
pip3 install natten==0.17.3+torch250cu124 -f https://shi-labs.com/natten/wheels/
```
2. You need to change code in `transformer.models.dinat.modeling_dinat.py` to surpress the warning of `natten` package:
> Tips: You can click `DinatModel` in `src/models/encoder/segmentation.py` to jump to the file.

turn
```python
if is_natten_available():
    from natten.functional import natten2dav, natten2dqkrpb
else:
    ...
```
to
```python
if is_natten_available():
    # from natten.functional import natten2dav, natten2dqkrpb
    from natten.functional import na2d_av 
    natten2dav = lambda attn, value, kernel_size, dilation: na2d_av(attn, value, kernel_size, dilation)
    from natten.functional import na2d_qk
    natten2dqkrpb  = lambda query, key, rpb, kernel_size, dilation: na2d_qk(query, key, kernel_size, dilation, rpb=rpb)
else:
    ...
```

## DeepSpeed

To properly install the package `mpi4py` required by DeepSpeed, you need to install `libmpich-dev` first, e.g.:
```bash
sudo apt install -y libmpich-dev
```

Alternatively, in case you don't have `sudo` permission, you can build from source:
```bash
# Download and build
wget https://github.com/pmodels/mpich/releases/download/v4.3.0b1/mpich-4.3.0b1.tar.gz
tar -xvf mpich-4.3.0b1.tar.gz && cd mpich-4.3.0b1
./configure --disable-fortran --prefix=$HOME/.local/mpich 2>&1 | tee c.txt
make 2>&1 | tee m.txt
make install 2>&1 | tee mi.txt
# Add installations to system paths
export PATH=$HOME/.local/mpich/bin:$PATH
export LD_LIBRARY_PATH=$HOME/.local/mpich/lib:$LD_LIBRARY_PATH
```

# Run Training Code

```bash
# Non-distributed training
python -m src.train_clean <CONFIG_FLAGS>
# Distributed training with DeepSpeed
deepspeed --module src.train_ds <CONFIG_FLAGS>
```

- You can use `--name <PRESET_NAME>` to load a configuration preset from YAML file.
- The scripts will create a preset file with default settings in configs/<PRESET_NAME>.yaml and exit for the first time.

# Run Inference Code

```bash
python -m src.inference --name <PRESET_NAME> --training_dir <TRAINING_OUTPUT_DIR> --ckpt_path <CHECKPOINT_PATH> <CONFIG_FLAGS>
```

- You need to provide the training output directory for checkpoint loading.
- Note that `ckpt_path` is **relative** to `training_dir`.
- The scripts will create a preset file with default settings in `configs/inference/<PRESET_NAME>.yaml` and exit for the first time.

### Assets Structure

```
configs/
└── <RUN_NAME>.yaml  # Main configuration
outputs/
└── <RUN_NAME>_<TIMESTAMP>/
    ├── checkpoint/
    ├── log/
    └── config.yaml  # Additional info
```