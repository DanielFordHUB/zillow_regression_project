import itertools
import seaborn as sns
import matplotlib.pyplot as plt



def plot_variable_pairs(df):
    '''
    Funtion used to quickly visualize variable pairs using lmplot
    '''
    quant_features = [col for col in df.columns if df[col].dtype != object]
    feature_combos = list(itertools.combinations(quant_features, 2))
    for combo in feature_combos:
        sns.lmplot(x=combo[0], y=combo[1], data=df, line_kws={'color': 'orange'})
        plt.show()



def plot_categorical_and_continuous_vars(df):
    '''
        function to quickly vizualize all combinations of catagorical and continous variables seperated into three columns based on barplot, boxplot, and swarm plot
        '''
    
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
    '''
    simple function to create joint plots
    '''
    sns.jointplot(x = x, y = y, data = df, kind = 'reg',
                  joint_kws = {'line_kws':{'color':'orange'}});