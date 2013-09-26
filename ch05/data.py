import os

# DATA_DIR = r"C:\pymlbook-data\ch05"
DATA_DIR = r"/media/sf_C/pymlbook-data/ch05"
CHART_DIR = os.path.join("..", "charts")

filtered = os.path.join(DATA_DIR, "filtered.tsv")
filtered_meta = os.path.join(DATA_DIR, "filtered-meta.json")

chosen = os.path.join(DATA_DIR, "chosen.tsv")
chosen_meta = os.path.join(DATA_DIR, "chosen-meta.json")
