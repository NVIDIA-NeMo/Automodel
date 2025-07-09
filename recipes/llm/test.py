from nemo_automodel.datasets.indexed_dataset import IndexedDataset  # <- note: datasets, not data

# common prefix – do NOT include ".bin"/".idx"
prefix = "/lustre/fsw/coreai_dlalgo_nemofw/dpykhtar/dclm/preprocessed/dclm_01_text_document"

ds = IndexedDataset(prefix)

print("Sequences:", len(ds))
tokens = ds[0]                       # numpy array of token-ids
print(tokens[:20])
