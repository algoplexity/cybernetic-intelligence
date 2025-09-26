--- Starting Month 2 Analysis: Compliance by Industry Sector ---
Step 1/4: Data prepared for industry analysis.
Step 2/4: Calculated average non-compliance per industry.
Step 3/4: Identified top 10 industries with highest average non-compliance.
Step 4/4: Generating final summary report.

--- Month 2: Top 10 High-Risk Industries (by Avg. Non-Compliant Criteria) ---
Analysis based on 737 single-lodger entities. Showing industries with 5 or more entities.
--------------------------------------------------------------------------------
                                        industry_desc  entity_count  \
83                   Land Development and Subdivision             5   
111                    Office Administrative Services             6   
147                  Other Social Assistance Services            11   
49                          Financial Asset Investing            86   
215                                           Unknown            33   
189                      Road and Bridge Construction             6   
24                                 Clothing Retailing             6   
108                Non-Residential Property Operators            15   
106                          Non-Depository Financing             8   
139  Other Machinery and Equipment Wholesaling n.e.c.             5   

     avg_non_compliant_criteria  
83                     1.000000  
111                    0.833333  
147                    0.818182  
49                     0.767442  
215                    0.696970  
189                    0.666667  
24                     0.666667  
108                    0.666667  
106                    0.625000  
139                    0.600000  
--------------------------------------------------------------------------------
--- Analysis Complete ---
/tmp/ipykernel_710/807058521.py:8: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.

For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.


  df_with_compliance_data['industry_desc'].fillna('Unknown', inplace=True)
