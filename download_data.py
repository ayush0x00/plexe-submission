import shutil
from pathlib import Path

import kagglehub

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"

# Download latest version (to kagglehub cache)
path = kagglehub.dataset_download("olistbr/brazilian-ecommerce")

DATA_DIR.mkdir(parents=True, exist_ok=True)
for f in Path(path).iterdir():
    dest = DATA_DIR / f.name
    if f.is_dir():
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(f, dest)
    else:
        shutil.copy2(f, dest)

print("Dataset copied to:", DATA_DIR)