import ezclimate.storage_tree as st

sst = st.SmallStorageTree(decision_times=[0, 15, 45, 85, 185, 285, 385])

sst.tree[385]
sst[385] # BaseStorageClass defines its own __getitem__

bst = st.BigStorageTree(subinterval_len=5, decision_times=[0, 15, 45, 85, 185, 285, 385])
bst[380] # time period that is not a decision time
bst[385]