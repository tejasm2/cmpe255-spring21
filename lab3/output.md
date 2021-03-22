| Experiement | Accuracy | Confusion Matrix | Comment |
|-------------|----------|------------------|---------|
| Baseline    | 0.6770833333333334 | [[114  16] [ 46  16]] |  |
| Solution 1   | 0.6927083333333334  | [[102  25] [ 34  31]] |  The accuracy improved when random state was 100 while splitting the data. |
| Solution 2   | 0.7239583333333334  | [[106  21] [ 32  33]] |  The accuracy improved when "Glucose" feature was added along with random state 100 while splitting data. |
| Solution 3   | 0.75  | [[111  16] [ 32  33]] |  The accuracy improved when Liblinear solver along with max iterations of 1000 were introduced and 'Pedigree' was added to the features with 'Glucose'.|
