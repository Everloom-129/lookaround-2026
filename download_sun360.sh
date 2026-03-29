# once
pip install -U gdown
# or: uv tool install gdown   # then use `gdown` from PATH

cd /mnt/sda/edward/data_lookaround

gdown \
  "https://drive.google.com/drive/folders/1ooaYwvNuFd-iEEcmOQHpLunJEmo7b4NM" \
  -O . \
  --folder \
  --continue \
  --remaining-ok