--- Starting Diagnostic Check for Submission Results ---

--- Diagnostic Check 1: Raw Value Counts from Master Register ---

Unique values in 'Submitted more than 6 months?':
Submitted more than 6 months?
With In 6       10467
NaN              4770
Outside of 6     4758
Name: count, dtype: int64

Unique values in 'Status':
Status
Published    14308
Draft         4961
Redraft        703
Hidden          23
Name: count, dtype: int64
--------------------------------------------------

--- Diagnostic Check 2: Manually Recalculating Late Submissions ---
Analyzed 15225 statements with valid timestamps.
Manual Recalculation Result: Found 4892 late submissions (32.1%).
--------------------------------------------------

--- Diagnostic Complete ---
