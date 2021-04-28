import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc


#categorical checker (check for object types in the dataset)
def cat_check(df):
    '''
    Checks for object dtypes in the dataframe and returns the unique values in each
    
    Args: df:DataFrame()
    Returns: df:DataFrame() - DataFrame containing only object types and number of unique values in them
    
    '''
    categorical = [index for index, i in enumerate(df.dtypes) if i == 'object']

    return df.iloc[:, categorical].nunique()


def feature_transform(df):
    
    '''
    Performs special feature transformation on the initial dataframe
    
    Args: df:DataFrame()
    Returns: df:DataFrame
    
    '''
    
    #clean special symbols from the features
    df['CAMEO_DEUG_2015'] = df['CAMEO_DEUG_2015'].replace('X', None)
    df['CAMEO_INTL_2015'] = df['CAMEO_INTL_2015'].replace('XX', None)
    df['CAMEO_DEU_2015'] = df['CAMEO_DEU_2015'].replace('XX', None)
    
    df['CAMEO_DEUG_2015'] = df['CAMEO_DEUG_2015'].astype('float')
    df['CAMEO_INTL_2015'] = df['CAMEO_INTL_2015'].astype('float')
    
    OST_WEST_KZ = {'W':0, 'O':1}
    
    df['OST_WEST_KZ_E'] = df.loc[:, 'OST_WEST_KZ'].map(OST_WEST_KZ)
    #extract year from datetime
    df['EINGEFUEGT_AM']=pd.to_datetime(df['EINGEFUEGT_AM']).dt.year
    
    df = df.drop(['Unnamed: 0', 'LNR','D19_LETZTER_KAUF_BRANCHE', 'OST_WEST_KZ', 'CAMEO_DEU_2015'  ], axis = 1)
    

    decades_dict = {1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3, 7: 3,
           8: 4, 9: 4, 10: 5, 11: 5, 12: 5, 13: 5, 14: 6,
           15: 6, 0: np.nan}
    df['PRAEGENDE_JUGENDJAHRE_DECADE'] = df['PRAEGENDE_JUGENDJAHRE'].map(decades_dict)
    print('Creating PRAEGENDE_JUGENDJAHRE_DECADE feature')
    
    
    movement_dict = {1: 0, 2: 1, 3: 0, 4: 1, 5: 0, 6: 1, 7: 1, 8: 0,
           9: 1, 10: 0, 11: 1, 12: 0, 13: 1, 14: 0, 15: 1, 0: np.nan}
    df['PRAEGENDE_JUGENDJAHRE_MOVEMENT'] = df['PRAEGENDE_JUGENDJAHRE'].map(movement_dict)
    
    print('Creating PRAEGENDE_JUGENDJAHRE_MOVEMENT feature')
    
    df.drop(['PRAEGENDE_JUGENDJAHRE'], axis = 1, inplace = True)
   

    return df

def unknown_unify(df, xls):
    '''
    Uses the xls attribute description file to locate Unknown values and replaces them with const -1
    
    Args: df:DataFrame(), xls:DataFrame()
    Returns: df:DataFrame
    
    '''
    
    #using the DIAs xls file lets save meanings that might indicate unknown values
    unknowns = xls['Meaning'].where(xls['Meaning'].str.contains('unknown')).value_counts().index
    
    #I will now create a list of all the unknown values for each attribute and replace them on my azdias and customers
    missing_unknowns = xls[xls['Meaning'].isin(unknowns)]

    for row in missing_unknowns.iterrows():
        missing_values = row[1]['Value']
        attribute = row[1]['Attribute']
        
        #dealing with columns that only exist in df
        if attribute not in df.columns:
            continue
        
        #dealing with strings or ints
        if isinstance(missing_values,int): 
            df[attribute].replace(missing_values, -1, inplace=True)
        elif isinstance(missing_values,str):
            eval("df[attribute].replace(["+missing_values+"], -1, inplace=True)")
            df[attribute].replace(missing_values, -1, inplace=True)
            
    return df

def predict_sub(X_train, y_train, model, X_test):
    
    '''
    Trains the final model and exports Submission.xls file for uploading to the Kaggle competition
    
    Args: 
        X_train: DataFrame() - Train data 
        y_train: DataFrame() - Train data labels
        model: sklearn.model - Final model to train
        X_test: string - Test data to predict for the Kaggle competition.
    
    '''

    model.fit(X_train, y_train)

    y_sub = model.predict_proba(X_test)[:, 1]

    submission = pd.DataFrame({'LNR':mailout_test['LNR'], 'RESPONSE': y_sub })
    submission.to_csv('submission_XGB.csv', index = False)

    
def plot_roc(model,X, y, model_name):
    
    '''
    Plots ROC curve for the imputed model.
    
    Args: 
        model: sklearn.model - model to evaluate
        X: DataFrame() - Train data 
        y: DataFrame() - Train data labels
        model_name: string - Name of the model.
    
    Returns: 
      
    '''

    X_train, X_test, y_train, y_test= train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y )
    model.fit(X_train, y_train)
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    y_score = model.predict_proba(X_test)[:, 1]
    n_classes = 2
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test, y_score)
        roc_auc[i] = auc(fpr[i], tpr[i])


    lw = 2
    ax  = sns.lineplot(x = fpr[0],y = tpr[0], lw=lw, label=model_name + '(AUC = %0.2f)' % roc_auc[0])

    ax = sns.lineplot(x = [0, 1], y = [0, 1], color='navy', lw=lw, linestyle='--')

    ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate')

    plt.title('ROC Curve')

     
    
