from pathlib import Path


def delete_dir(dir: Path):
    dir = Path(dir)
    if not dir.exists():
        return
    if not dir.is_dir():
        return

    for file in dir.iterdir():
        if file.is_dir():
            delete_dir(file)
        else:
            file.unlink()
    dir.rmdir()


if __name__ == "__main__":
    BASE_DIR = Path("outputs")
    for dir in BASE_DIR.iterdir():
        if not dir.is_dir():
            continue
        ckpt_dir = dir / "checkpoint"
        to_delete = not ckpt_dir.exists()
        if ckpt_dir.exists():
            to_delete = not any(ckpt_dir.glob("*.pt"))
        if to_delete:
            print(f"Deleting {dir}")
            delete_dir(dir)
