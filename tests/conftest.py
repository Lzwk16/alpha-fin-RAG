import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # the models are cached; skip the Hub check
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
