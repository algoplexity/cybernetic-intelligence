--- Starting FINAL Month 2 Analysis: Late and Non-Publishable Submissions (Corrected) ---
Step 1/4: Data prepared for analyzing 4198 single-lodger entities.
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
Cell In[18], line 12
      8 print(f"Step 1/4: Data prepared for analyzing {total_entities} single-lodger entities.")
     10 # --- 2. LATE SUBMISSIONS: Use the Manual Recalculation Method ---
     11 # This is our validated ground truth.
---> 12 df_validation = df_single_lodgers[['abn', 'Period end date', 'Submitted']].copy()
     13 df_validation['Period end date'] = pd.to_datetime(df_validation['Period end date'], errors='coerce')
     14 df_validation['Submitted'] = pd.to_datetime(df_validation['Submitted'], errors='coerce')

File /opt/conda/lib/python3.12/site-packages/pandas/core/frame.py:4113, in DataFrame.__getitem__(self, key)
   4111     if is_iterator(key):
   4112         key = list(key)
-> 4113     indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4115 # take() does not accept boolean indexers
   4116 if getattr(indexer, "dtype", None) == bool:

File /opt/conda/lib/python3.12/site-packages/pandas/core/indexes/base.py:6212, in Index._get_indexer_strict(self, key, axis_name)
   6209 else:
   6210     keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
-> 6212 self._raise_if_missing(keyarr, indexer, axis_name)
   6214 keyarr = self.take(indexer)
   6215 if isinstance(key, Index):
   6216     # GH 42790 - Preserve name from an Index

File /opt/conda/lib/python3.12/site-packages/pandas/core/indexes/base.py:6264, in Index._raise_if_missing(self, key, indexer, axis_name)
   6261     raise KeyError(f"None of [{key}] are in the [{axis_name}]")
   6263 not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 6264 raise KeyError(f"{not_found} not in index")

KeyError: "['Period end date'] not in index"
