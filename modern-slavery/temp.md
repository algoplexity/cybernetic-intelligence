--- Starting Month 2 Analysis: Late and Non-Publishable Submissions ---
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
File /opt/conda/lib/python3.12/site-packages/pandas/core/indexes/base.py:3812, in Index.get_loc(self, key)
   3811 try:
-> 3812     return self._engine.get_loc(casted_key)
   3813 except KeyError as err:

File pandas/_libs/index.pyx:167, in pandas._libs.index.IndexEngine.get_loc()

File pandas/_libs/index.pyx:196, in pandas._libs.index.IndexEngine.get_loc()

File pandas/_libs/hashtable_class_helper.pxi:7088, in pandas._libs.hashtable.PyObjectHashTable.get_item()

File pandas/_libs/hashtable_class_helper.pxi:7096, in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'Submitted more than 6 months?'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
Cell In[14], line 9
      1 print("\n--- Starting Month 2 Analysis: Late and Non-Publishable Submissions ---")
      3 # --- 1. Prepare the Data ---
      4 # The df_single_lodgers DataFrame from the V2 deliverable is our source.
      5 # Let's ensure the relevant columns are clean and ready for analysis.
      6 # We'll work with the full cohort of 4198 single lodgers.
      7 
      8 # Handle potential missing values in key columns
----> 9 df_single_lodgers['Submitted more than 6 months?'].fillna('Unknown', inplace=True)
     10 df_single_lodgers['Status'].fillna('Unknown', inplace=True)
     11 print(f"Step 1/4: Data prepared for analyzing {len(df_single_lodgers)} single-lodger entities.")

File /opt/conda/lib/python3.12/site-packages/pandas/core/frame.py:4107, in DataFrame.__getitem__(self, key)
   4105 if self.columns.nlevels > 1:
   4106     return self._getitem_multilevel(key)
-> 4107 indexer = self.columns.get_loc(key)
   4108 if is_integer(indexer):
   4109     indexer = [indexer]

File /opt/conda/lib/python3.12/site-packages/pandas/core/indexes/base.py:3819, in Index.get_loc(self, key)
   3814     if isinstance(casted_key, slice) or (
   3815         isinstance(casted_key, abc.Iterable)
   3816         and any(isinstance(x, slice) for x in casted_key)
   3817     ):
   3818         raise InvalidIndexError(key)
-> 3819     raise KeyError(key) from err
   3820 except TypeError:
   3821     # If we have a listlike key, _check_indexing_error will raise
   3822     #  InvalidIndexError. Otherwise we fall through and re-raise
   3823     #  the TypeError.
   3824     self._check_indexing_error(key)

KeyError: 'Submitted more than 6 months?'
