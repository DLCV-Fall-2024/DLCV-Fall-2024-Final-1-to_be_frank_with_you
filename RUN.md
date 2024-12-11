# Download Dataset

```bash
python -m utils.dataset
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