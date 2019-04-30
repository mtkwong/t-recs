These files contain the top-k recommendations for each user.
The first column is the user ID (uid), and the subsequent columns, in order, represent the ranked recommendations (i.e. iid001 is the #1 recommended business, iid002 is #2, and so on).

The filenames are formatted as follows: top{n}_{algo}_{rating used}.csv
1. n = number of top recommendations given per user
2. algo = the algorithm used
    - "svd" = base SVD with default parameters;
    - "svd_opt" = optimized SVD with the following parameters: lr_bu=0.005, lr_bi=0.005, lr_pu=0.005, lr_qi=0.001, reg_bu=0.05, reg_bi=0.02, reg_pu=0.05, reg_qi=0.05
3. rating used = the rating column used for the training. 
    - raw: raw rating
    - time: timestamp-normalized rating
    - mean: mean-centered rating
    - meantime: mean-centered AND timestamp-normalized rating