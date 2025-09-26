--- Starting Month 3 Analysis: Systemic Risk Identification ---
ERROR: A required file or sheet could not be loaded. Details: Worksheet named 'associates' not found
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
Cell In[21], line 8
      5 # --- 1. Load All Necessary Data Sources ---
      6 try:
      7     # Load the associates data from the two source files
----> 8     assoc_non_lodger_df = pd.read_excel('ato_tax_transparency_non_lodger.xlsx', sheet_name='associates')
      9     assoc_single_lodger_df = pd.read_excel('lodge_once_cont.xlsx', sheet_name='associates')
     11     # Load our definitive lists of at-risk entities from the final Month 1 deliverable

File /opt/conda/lib/python3.12/site-packages/pandas/io/excel/_base.py:508, in read_excel(io, sheet_name, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skiprows, nrows, na_values, keep_default_na, na_filter, verbose, parse_dates, date_parser, date_format, thousands, decimal, comment, skipfooter, storage_options, dtype_backend, engine_kwargs)
    502     raise ValueError(
    503         "Engine should not be specified when passing "
    504         "an ExcelFile - ExcelFile already has the engine set"
    505     )
    507 try:
--> 508     data = io.parse(
    509         sheet_name=sheet_name,
    510         header=header,
    511         names=names,
    512         index_col=index_col,
    513         usecols=usecols,
    514         dtype=dtype,
    515         converters=converters,
    516         true_values=true_values,
    517         false_values=false_values,
    518         skiprows=skiprows,
    519         nrows=nrows,
    520         na_values=na_values,
    521         keep_default_na=keep_default_na,
    522         na_filter=na_filter,
    523         verbose=verbose,
    524         parse_dates=parse_dates,
    525         date_parser=date_parser,
    526         date_format=date_format,
    527         thousands=thousands,
    528         decimal=decimal,
    529         comment=comment,
    530         skipfooter=skipfooter,
    531         dtype_backend=dtype_backend,
    532     )
    533 finally:
    534     # make sure to close opened file handles
    535     if should_close:

File /opt/conda/lib/python3.12/site-packages/pandas/io/excel/_base.py:1616, in ExcelFile.parse(self, sheet_name, header, names, index_col, usecols, converters, true_values, false_values, skiprows, nrows, na_values, parse_dates, date_parser, date_format, thousands, comment, skipfooter, dtype_backend, **kwds)
   1576 def parse(
   1577     self,
   1578     sheet_name: str | int | list[int] | list[str] | None = 0,
   (...)
   1596     **kwds,
   1597 ) -> DataFrame | dict[str, DataFrame] | dict[int, DataFrame]:
   1598     """
   1599     Parse specified sheet(s) into a DataFrame.
   1600 
   (...)
   1614     >>> file.parse()  # doctest: +SKIP
   1615     """
-> 1616     return self._reader.parse(
   1617         sheet_name=sheet_name,
   1618         header=header,
   1619         names=names,
   1620         index_col=index_col,
   1621         usecols=usecols,
   1622         converters=converters,
   1623         true_values=true_values,
   1624         false_values=false_values,
   1625         skiprows=skiprows,
   1626         nrows=nrows,
   1627         na_values=na_values,
   1628         parse_dates=parse_dates,
   1629         date_parser=date_parser,
   1630         date_format=date_format,
   1631         thousands=thousands,
   1632         comment=comment,
   1633         skipfooter=skipfooter,
   1634         dtype_backend=dtype_backend,
   1635         **kwds,
   1636     )

File /opt/conda/lib/python3.12/site-packages/pandas/io/excel/_base.py:773, in BaseExcelReader.parse(self, sheet_name, header, names, index_col, usecols, dtype, true_values, false_values, skiprows, nrows, na_values, verbose, parse_dates, date_parser, date_format, thousands, decimal, comment, skipfooter, dtype_backend, **kwds)
    770     print(f"Reading sheet {asheetname}")
    772 if isinstance(asheetname, str):
--> 773     sheet = self.get_sheet_by_name(asheetname)
    774 else:  # assume an integer if not a string
    775     sheet = self.get_sheet_by_index(asheetname)

File /opt/conda/lib/python3.12/site-packages/pandas/io/excel/_openpyxl.py:582, in OpenpyxlReader.get_sheet_by_name(self, name)
    581 def get_sheet_by_name(self, name: str):
--> 582     self.raise_if_bad_sheet_by_name(name)
    583     return self.book[name]

File /opt/conda/lib/python3.12/site-packages/pandas/io/excel/_base.py:624, in BaseExcelReader.raise_if_bad_sheet_by_name(self, name)
    622 def raise_if_bad_sheet_by_name(self, name: str) -> None:
    623     if name not in self.sheet_names:
--> 624         raise ValueError(f"Worksheet named '{name}' not found")

ValueError: Worksheet named 'associates' not found
