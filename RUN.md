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


# Run Training Code

```bash
# Only creates config file and assets in the first run
python -m src.train_clean <RUN_NAME>
```

### Assets Structure

```
configs/
└── <RUN_NAME>.yaml  # Main configuration
outputs/
└── <RUN_NAME>_<TIMESTAMP>/
    ├── checkpoint/
    ├── log/
    └── config.yaml  # Additional info
...
```