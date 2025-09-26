--- Starting Month 2 Analysis: Late and Non-Publishable Submissions (Final Attempt) ---
Step 1/4: Data prepared for analyzing 4198 single-lodger entities.
Step 2/4: Calculated the number of late submissions.
Step 3/4: Calculated the number of non-publishable submissions.
Step 4/4: Generating final summary report.

--- Month 2: Submission Timeliness and Outcome Summary ---
Analysis based on all 4198 single-lodger entities.
-----------------------------------------------------------
Late Submissions (Over 6 months): 0 entities (0.0%)
Non-Publishable Submissions:      0 entities (0.0%)
-----------------------------------------------------------
--- Analysis Complete ---
/tmp/ipykernel_710/1933720191.py:12: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.

For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.


  df_single_lodgers['Submitted more than 6 months?'].fillna('Unknown', inplace=True)
/tmp/ipykernel_710/1933720191.py:18: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.

For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.


  df_single_lodgers['Status'].fillna('Unknown', inplace=True)
