# Recommender Systems

### Dataset
We used the movielens 20m (1000 users) and 150k ratings. Eigenvalues = 996 (Reference for calculation of errors and time taken to predict)

## Assumptions
### Neighbours considered in collab filtering

After running empirical tests with multiple k values, all being multiples of 15, it was observed that 60 yielded the most optimal solution. Thus k = 60 was chosen to be the final nearest neighbours valuation.

### Rows and columns in CUR 
With reference to class slides, rank of user rating matrix is calculated after which, 4 times the value is taken to select the rows and columns.

## General Specs
Packages used primarily:
-   **numpy:** NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.
    
-   **scipy.sparse:** a better alternative to employing to sparse matrices than what numpy has to offer.
    

-   **csc_matrix, csr_matrix:** sparse representation of matrices can be done in either of two ways – by leading column indices (csc_matrix) or leading row indices (csr_matrix)
    
-   **linalg:** helps employ mathematical and matrix operations that correspond to eigen value or simply any general matrix application scenario.
    

-   **pandas, csv:** data analysis and manipulation
    
-   **sklearn.metrics:** imported mean square error for root mean square error calculations wrt collaborative, SVD, and CUR recommended models.
    

* **math:** for calculating basic mathematical expressions. In our models, we extensively used the math module to enable smoother square root functionality, especially in the case of the calculation of RMSE.

## Formulae
### Splitting into test and train
The original dataset is intended to be split into a train dataset and a test set. 10% non-zero values are chosen at random in each row, and turned to zero. Due to the nature of our dataset, every occurrence of zero can be safely treated as an empty entry due to the lowest rating by any user being 0.5.

The models are run through this train dataset and appropriate predictions are made. These are then compared with those in test (which comprises only those values that were omitted from train).

Appropriate error calculations follow once the model predicts using the train set.

### Collaborative 
After the train/test split, the user vectors are mean adjusted to account for generous and strict (at rating) users.

Similarity between users and items are calculated by employing cosine similarity.

In the case of baseline, we first define similarity between items (s(i, j) between items i and j) using Pearson coefficient.

$s(i, j) =  \frac{\sum_{u \in U}(R_{u,i} - \bar{R}_u)(R_{u,j} - \bar{R}_u)}{\sqrt{\sum_{u \in U}(R_{u,i} - \bar{R}_u)^2}\sqrt{\sum_{u \in U}(R_{u,j} - \bar{R}_u)^2}}$

Then, we select the k nearest neighbours N(i; x) after running empirical tests on multiple possible k values and finalising on an optimal one.

Rating r<sub>xi</sub> is estimated as the weighted average:
	
$r_{xi} = b_{xi} + \frac{\sum_{j \in N(i;x)}s_{ij}.(r_{xj}-b_{xj})}{\sum_{j \in N(i;x)}s_{ij}}$
$b_{xi}= μ + b_{x}+b_{i}$

where

$b_xi$ = baseline estimate for $r_xi$

$μ$ = overall mean movie rating

$b_x$= rating deviation of user x = (avg. rating of user x) – μ

$b_i$ = rating deviation of movie i = (avg. rating of movie i) – μ

The weighted average of the ratings is taken on the basis of similarity scores, after taking into consideration the nearest neighbours, and that yields the respective prediction for that case.

### Singular Value Decomposition (SVD) and CUR
After the train/test split, the user vectors are mean adjusted to account for generous and strict (at rating) users. This is achieved by calculating the user bias, which is simply the row average over all non-zero elements in every user vector.

After obtaining matrices U, V, and S, query matrices are multiplied to SVT. A two-dimensional matrix of all such user values ideally yields the entire prediction set.

For 90% energy, the sum of squares of the matrix S are used to compute the extent of dimensionality reduction.

CUR follows slightly similar steps. In the case of CUR, the C and R matrices are drawn up using random distribution of the columns and rows of the user ratings matrix. Then matrix W is found, which is the intersection of C and R. Matrix U is the Moore-Penrose pseudoinverse of matrix W. Matrices C, U, and R are analogous to U, S, and V in SVD.

## Error calculations
**RMSE**
Standard error calculation between Predicted values and Actual Values.
 
**Precision on top k**
Default k value is taken as 25 and actual rating above 3.5 is taken as relevant item to the user.

**Spearman rank correlation**
The rank of the predictions are calculated for every user and subtracted with actual rank of the predicted movie from test dataset and squared. This value is then substituted in the formula to get the correlation. It is always less than 1.

**Time**
It was calculated after loading matrix into RAM till output is printed.

## Observations
Recommender System Model | RMSE | Precision on top k | Spearman Rank Correlation | Time taken for prediction
--- | --- | --- |--- | --- 
Collaborative | 0.26 | 0.9 | 0.992 | 1.28/value
Collab + baseline | 0.25 | 0.93 | 0.993 | 2.69/value
SVD | 0 | 1 | 1 | 81s
SVD + 90% energy retained | 0.9928 | 0.96 | 0.992 | 53s
CUR | 0.992 | 0.94 | 0.99 | 42s
CUR + 90% energy retained | 0.9939 | 0.96 | 0.991 | 33s
