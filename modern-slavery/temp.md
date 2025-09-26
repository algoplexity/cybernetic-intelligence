--- Starting FINAL Month 3 Analysis: Systemic Risk Identification ---
Step 1/5: All source data for associates and at-risk cohorts loaded successfully.
Step 2/5: Consolidated 15958 associate records and 2379 at-risk ABNs.
Step 3/5: Identified 1 links between associates and at-risk entities.
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
Cell In[23], line 44
     37 at_risk_associates_df['assoc_fmly_nm'] = at_risk_associates_df['assoc_fmly_nm'].fillna('')
     39 at_risk_associates_df['associate_identifier'] = at_risk_associates_df.apply(
     40     lambda row: row['assoc_org_nm'] if row['assoc_org_nm'] != '' else f"{row['assoc_gvn_nm']} {row['assoc_fmly_nm']}".strip(),
     41     axis=1
     42 )
---> 44 systemic_risk_summary = at_risk_associates_df.groupby('associate_identifier').agg(
     45     entity_count=('abn', 'nunique'),
     46     roles=('rltnshp_cd', lambda x: ', '.join(x.unique()))
     47 ).reset_index()
     49 high_risk_associates = systemic_risk_summary[systemic_risk_summary['entity_count'] > 1].sort_values(
     50     by='entity_count', ascending=False
     51 )
     52 print("Step 4/5: Calculated and ranked associates by number of connections to at-risk entities.")

File /opt/conda/lib/python3.12/site-packages/pandas/core/groupby/generic.py:1432, in DataFrameGroupBy.aggregate(self, func, engine, engine_kwargs, *args, **kwargs)
   1429     kwargs["engine_kwargs"] = engine_kwargs
   1431 op = GroupByApply(self, func, args=args, kwargs=kwargs)
-> 1432 result = op.agg()
   1433 if not is_dict_like(func) and result is not None:
   1434     # GH #52849
   1435     if not self.as_index and is_list_like(func):

File /opt/conda/lib/python3.12/site-packages/pandas/core/apply.py:190, in Apply.agg(self)
    187     return self.apply_str()
    189 if is_dict_like(func):
--> 190     return self.agg_dict_like()
    191 elif is_list_like(func):
    192     # we require a list, but not a 'str'
    193     return self.agg_list_like()

File /opt/conda/lib/python3.12/site-packages/pandas/core/apply.py:423, in Apply.agg_dict_like(self)
    415 def agg_dict_like(self) -> DataFrame | Series:
    416     """
    417     Compute aggregation in the case of a dict-like argument.
    418 
   (...)
    421     Result of aggregation.
    422     """
--> 423     return self.agg_or_apply_dict_like(op_name="agg")

File /opt/conda/lib/python3.12/site-packages/pandas/core/apply.py:1603, in GroupByApply.agg_or_apply_dict_like(self, op_name)
   1598     kwargs.update({"engine": engine, "engine_kwargs": engine_kwargs})
   1600 with com.temp_setattr(
   1601     obj, "as_index", True, condition=hasattr(obj, "as_index")
   1602 ):
-> 1603     result_index, result_data = self.compute_dict_like(
   1604         op_name, selected_obj, selection, kwargs
   1605     )
   1606 result = self.wrap_results_dict_like(selected_obj, result_index, result_data)
   1607 return result

File /opt/conda/lib/python3.12/site-packages/pandas/core/apply.py:497, in Apply.compute_dict_like(self, op_name, selected_obj, selection, kwargs)
    493         results += key_data
    494 else:
    495     # key used for column selection and output
    496     results = [
--> 497         getattr(obj._gotitem(key, ndim=1), op_name)(how, **kwargs)
    498         for key, how in func.items()
    499     ]
    500     keys = list(func.keys())
    502 return keys, results

File /opt/conda/lib/python3.12/site-packages/pandas/core/groupby/generic.py:257, in SeriesGroupBy.aggregate(self, func, engine, engine_kwargs, *args, **kwargs)
    255 kwargs["engine"] = engine
    256 kwargs["engine_kwargs"] = engine_kwargs
--> 257 ret = self._aggregate_multiple_funcs(func, *args, **kwargs)
    258 if relabeling:
    259     # columns is not narrowed by mypy from relabeling flag
    260     assert columns is not None  # for mypy

File /opt/conda/lib/python3.12/site-packages/pandas/core/groupby/generic.py:362, in SeriesGroupBy._aggregate_multiple_funcs(self, arg, *args, **kwargs)
    360     for idx, (name, func) in enumerate(arg):
    361         key = base.OutputKey(label=name, position=idx)
--> 362         results[key] = self.aggregate(func, *args, **kwargs)
    364 if any(isinstance(x, DataFrame) for x in results.values()):
    365     from pandas import concat

File /opt/conda/lib/python3.12/site-packages/pandas/core/groupby/generic.py:294, in SeriesGroupBy.aggregate(self, func, engine, engine_kwargs, *args, **kwargs)
    291     return self._python_agg_general(func, *args, **kwargs)
    293 try:
--> 294     return self._python_agg_general(func, *args, **kwargs)
    295 except KeyError:
    296     # KeyError raised in test_groupby.test_basic is bc the func does
    297     #  a dictionary lookup on group.name, but group name is not
    298     #  pinned in _python_agg_general, only in _aggregate_named
    299     result = self._aggregate_named(func, *args, **kwargs)

File /opt/conda/lib/python3.12/site-packages/pandas/core/groupby/generic.py:327, in SeriesGroupBy._python_agg_general(self, func, *args, **kwargs)
    324 f = lambda x: func(x, *args, **kwargs)
    326 obj = self._obj_with_exclusions
--> 327 result = self._grouper.agg_series(obj, f)
    328 res = obj._constructor(result, name=obj.name)
    329 return self._wrap_aggregated_output(res)

File /opt/conda/lib/python3.12/site-packages/pandas/core/groupby/ops.py:864, in BaseGrouper.agg_series(self, obj, func, preserve_dtype)
    857 if not isinstance(obj._values, np.ndarray):
    858     # we can preserve a little bit more aggressively with EA dtype
    859     #  because maybe_cast_pointwise_result will do a try/except
    860     #  with _from_sequence.  NB we are assuming here that _from_sequence
    861     #  is sufficiently strict that it casts appropriately.
    862     preserve_dtype = True
--> 864 result = self._aggregate_series_pure_python(obj, func)
    866 npvalues = lib.maybe_convert_objects(result, try_float=False)
    867 if preserve_dtype:

File /opt/conda/lib/python3.12/site-packages/pandas/core/groupby/ops.py:885, in BaseGrouper._aggregate_series_pure_python(self, obj, func)
    882 splitter = self._get_splitter(obj, axis=0)
    884 for i, group in enumerate(splitter):
--> 885     res = func(group)
    886     res = extract_result(res)
    888     if not initialized:
    889         # We only do this validation on the first iteration

File /opt/conda/lib/python3.12/site-packages/pandas/core/groupby/generic.py:324, in SeriesGroupBy._python_agg_general.<locals>.<lambda>(x)
    322     alias = com._builtin_table_alias[func]
    323     warn_alias_replacement(self, orig_func, alias)
--> 324 f = lambda x: func(x, *args, **kwargs)
    326 obj = self._obj_with_exclusions
    327 result = self._grouper.agg_series(obj, f)

Cell In[23], line 46, in <lambda>(x)
     37 at_risk_associates_df['assoc_fmly_nm'] = at_risk_associates_df['assoc_fmly_nm'].fillna('')
     39 at_risk_associates_df['associate_identifier'] = at_risk_associates_df.apply(
     40     lambda row: row['assoc_org_nm'] if row['assoc_org_nm'] != '' else f"{row['assoc_gvn_nm']} {row['assoc_fmly_nm']}".strip(),
     41     axis=1
     42 )
     44 systemic_risk_summary = at_risk_associates_df.groupby('associate_identifier').agg(
     45     entity_count=('abn', 'nunique'),
---> 46     roles=('rltnshp_cd', lambda x: ', '.join(x.unique()))
     47 ).reset_index()
     49 high_risk_associates = systemic_risk_summary[systemic_risk_summary['entity_count'] > 1].sort_values(
     50     by='entity_count', ascending=False
     51 )
     52 print("Step 4/5: Calculated and ranked associates by number of connections to at-risk entities.")

TypeError: sequence item 0: expected str instance, float found
