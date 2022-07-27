import itertools
import seaborn as sns
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt



def plot_variable_pairs(df):
    quant_features = [col for col in df.columns if df[col].dtype != object]
    feature_combos = list(itertools.combinations(quant_features, 2))
    for combo in feature_combos:
        sns.lmplot(x=combo[0], y=combo[1], data=df, line_kws={'color': 'orange'})
        plt.show()



def plot_categorical_and_continuous_vars(df):    
    
    # seperate variable
    categ_vars = [col for col in df.columns if (df[col].dtype == 'object') or (len(df[col].unique()) < 10)]
    cont_vars = [col for col in df.columns if (col not in categ_vars)]
    
    
    for cont_var in cont_vars:
        for categ_var in categ_vars:

            plt.figure(figsize=(30,10))
            
            # barplot of average values
            plt.subplot(131)
            sns.barplot(data=df,
                        x=categ_var,
                        y=cont_var)
            plt.axhline(df[cont_var].mean(), 
                        ls='--', 
                        color='black')
            plt.title(f'{cont_var} by {categ_var}', fontsize=14)
            
            # box plot of distributions
            plt.subplot(132)
            sns.boxplot(data=df,
                          x=categ_var,
                          y=cont_var)
            
            # swarmplot of distributions
            
            # for larger datasets, use a sample of n=1000
            if len(df) > 1000:
                train_sample = df.sample(1000)

                plt.subplot(133)
                sns.swarmplot(x=categ_var,
                              y=cont_var,
                              data=train_sample)
            
            # for smaller datasets, plot all data
            else:
                plt.subplot(133)
                sns.swarmplot(x=categ_var,
                              y=cont_var,
                              data=df)
            plt.show()
            
            
def joint(x, y, df):
    sns.jointplot(x = x, y = y, data = df, kind = 'reg',
                  joint_kws = {'line_kws':{'color':'orange'}});


def significance_test(p):
    '''
    Assumes an alpha of .05. Takes in a value, p, and returns a string stating whether or not that p value indicates sufficient evidence
    to reject a null hypothesis.
    '''
    α = 0.05
    if p < α:
        print("Sufficient evidence -> Reject the null hypothesis.")
    else:
        print("Insufficient evidence -> Fail to reject the null hypothesis.")


def value_correlations(df):
    '''
    This function takes in the dataframe and uses pandas and seaborn to create a
    ordered list and heatmap of the correlations between the various quantitative feeatures and the target. 
    '''
    # create a dataframe of correlation values, sorted in descending order
    corr = pd.DataFrame(df.corr().abs().tax_value).sort_values(by='tax_value', ascending=False)
    # rename the correlation column
    corr.columns = ['correlation (abs)']
    # establish figure size
    plt.figure(figsize=(10,10))
    # creat the heatmap using the correlation dataframe created above
    sns.heatmap(corr, annot=True, cmap="mako")
    # establish a plot title
    plt.title('Features\' Correlation with Value')
    # display the plot
    plt.show()



