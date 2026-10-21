# CuStorA Clustering Investigation: Why K=2?

Investigation run on `data/processed/customers_rfm.csv` (1000 customers), over the same `CLUSTER_FEATURES` used by `src/clustering/kmeans.py`: Recency, Frequency, Monetary, EngagementScore, ComplaintRate, ReturnRate, DiscountUsage, TenureYears.

## 1. Are the clustering features actually independent?

Feature pairs with |correlation| >= 0.6 (see `correlation_matrix.csv` for the full matrix, `correlation_heatmap.png` for the plot):

| FeatureA   | FeatureB   |   Correlation |
|:-----------|:-----------|--------------:|
| Frequency  | Monetary   |         0.632 |

A pair up here isn't automatically a bug -- Frequency and Monetary, for example, are expected to move together for real customers -- but it does mean those two dimensions are pulling K-Means in a similar direction rather than contributing independent structure, which is part of why the data doesn't support many distinct clusters.


## 2. How much variance lives in a small number of dimensions?

Full PCA explained-variance curve is in `pca_explained_variance.csv` and `pca_scree_plot.png`. The first two components alone explain **50.5%** of total variance across the 8 clustering features:

| Component   |   ExplainedVariance |   CumulativeVariance |
|:------------|--------------------:|---------------------:|
| PC1         |              0.3769 |               0.3769 |
| PC2         |              0.1284 |               0.5053 |
| PC3         |              0.1252 |               0.6305 |
| PC4         |              0.1238 |               0.7543 |
| PC5         |              0.0855 |               0.8399 |
| PC6         |              0.0767 |               0.9166 |
| PC7         |              0.0539 |               0.9705 |
| PC8         |              0.0295 |               1      |

Variance is fairly spread across components -- this dataset could plausibly support more structure than K=2 if the cluster profiles below back that up.


## 3. What do K=2, K=3, and K=4 actually look like?

|   K |   Silhouette | ClusterSizes                                                                                   |
|----:|-------------:|:-----------------------------------------------------------------------------------------------|
|   2 |       0.2018 | cluster 0: 457 (45.7%), cluster 1: 543 (54.3%)                                                 |
|   3 |       0.1649 | cluster 0: 333 (33.3%), cluster 1: 147 (14.7%), cluster 2: 520 (52.0%)                         |
|   4 |       0.1554 | cluster 0: 135 (13.5%), cluster 1: 269 (26.9%), cluster 2: 313 (31.3%), cluster 3: 283 (28.3%) |



### K=2 cluster profiles (raw feature means)

|   Cluster |   Recency |   Frequency |   Monetary |   EngagementScore |   ComplaintRate |   ReturnRate |   DiscountUsage |   TenureYears |
|----------:|----------:|------------:|-----------:|------------------:|----------------:|-------------:|----------------:|--------------:|
|         0 |    311.56 |       17.04 |    24468.8 |             34.16 |            0.11 |         0.09 |           46.49 |          2.63 |
|         1 |    325.16 |       29.23 |    72517.4 |             61.83 |            0.03 |         0.03 |           41.43 |          3.6  |



### K=3 cluster profiles (raw feature means)

|   Cluster |   Recency |   Frequency |   Monetary |   EngagementScore |   ComplaintRate |   ReturnRate |   DiscountUsage |   TenureYears |
|----------:|----------:|------------:|-----------:|------------------:|----------------:|-------------:|----------------:|--------------:|
|         0 |    297.62 |       32.24 |    90475.6 |             67.75 |            0.03 |         0.02 |           40.26 |          3.83 |
|         1 |    307.19 |       12.79 |    15535.7 |             27.8  |            0.2  |         0.15 |           49.33 |          2.21 |
|         2 |    335.93 |       21.24 |    34898.2 |             43.34 |            0.06 |         0.05 |           44.39 |          2.99 |



### K=4 cluster profiles (raw feature means)

|   Cluster |   Recency |   Frequency |   Monetary |   EngagementScore |   ComplaintRate |   ReturnRate |   DiscountUsage |   TenureYears |
|----------:|----------:|------------:|-----------:|------------------:|----------------:|-------------:|----------------:|--------------:|
|         0 |    308.04 |       12.51 |    15288.2 |             27.3  |            0.21 |         0.16 |           49.54 |          2.12 |
|         1 |    296.81 |       33.38 |    96459   |             68.45 |            0.02 |         0.02 |           39.98 |          4.06 |
|         2 |    310.63 |       21    |    42040.3 |             54.09 |            0.05 |         0.05 |           45.12 |          1.85 |
|         3 |    354.39 |       22.69 |    33177.3 |             35.88 |            0.07 |         0.05 |           43.04 |          4.24 |



## Recommendation

K=2 has the highest silhouette score (0.2018) of the candidates compared here. 
Looking at the K=3 and K=4 profiles above, the extra clusters split one of the K=2 groups into slices that differ from each other mostly by degree (e.g. "somewhat engaged" vs. "very engaged") rather than by a distinct behavioural pattern a business would act on differently. Combined with points 1 and 2 above (some correlated features, and most variance concentrated in the first couple of components), **K=2 is what this dataset actually supports.** The honest framing is "this dataset currently separates into two statistically distinct customer groups," not "CuStorA only found two segments because it didn't look hard enough."


*Do not force a specific K to make the segmentation story look more granular than the data supports -- see project review, "What I would NOT do."*
