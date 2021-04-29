# Customer Segmetation Report and Mailout Campaign prediction

### 1. Project Overview:

The project is divided in two parts. First part uses unsupervised learning techniques to identified common clusters between the general population of Germany and the current customers of the company. The goal is to extract parts of the general population which is more likely to become a customer. The second part uses supervised learning to train model and predict data collected from mailout campaign.

### 2. Data

-  **Udacity_AZDIAS_052018.csv** : Demographics data for the general population of Germany; 891 211 persons (rows) x 366 features (columns).

-  **Udacity_CUSTOMERS_052018.csv** : Demographics data for customers of a mail-order company; 191 652 persons (rows) x 369 features (columns).

-  **Udacity_MAILOUT_052018_TRAIN.csv** : Demographics data for individuals who were targets of a marketing campaign; 42 982 persons (rows) x 367 (columns).

-  **Udacity_MAILOUT_052018_TEST.csv** : Demographics data for individuals who were targets of a marketing campaign; 42 833 persons (rows) x 366 (columns).


### 3. Files:

-  Arvato Project Workbook.ipynb 

-  utils.py (file with custom functions)

### 4. Build with:

-  Python 3.6, Jupyter Notebook

-  numpy, pandas, sklearn, xgboost

-  matplotlib, seaborn (visualization)

-  PCA, k-means clustering, XGBClassfier

### 5. Results:

- #### Part 1 Unsupervised learning:

- PCA components

- Elbow Curve

- Cluster distribution

- #### Part 2 Supervised learning:

- Initial models test

- Optimized model ROC Curve

### 6. Licensing, Authors, Acknowledgements:

Thanks to Arvato and Udacity for providing the oportunity to work on real life case.