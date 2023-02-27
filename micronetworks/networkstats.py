"""
Created on 12-01-2022
@author: Pedro J. Torres
"""
import pandas as pd
import numpy as np
import scipy
import os

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.pyplot import figure, text


from scipy.stats import pearsonr
from scipy.stats import spearmanr
from skbio.stats.composition import multiplicative_replacement, clr
from statsmodels.stats.multitest import multipletests
from sklearn import covariance
from sklearn.covariance import GraphicalLasso

import networkx as nx
import community as community_louvain

def mr_clr(df):
    """
    Calculate the center log ratio (log (abs(x)/g(m)))
    x is the abundance/read count of a given feature and
    g(m) is the geometric mean.
    df -- dataframe with columns as variables and rows as observations.
    """
    # make sure there are no NAs in data frame
    df = df.fillna(0)
    #replace all 0 entries with a small non-zero number
    clr_np = clr(multiplicative_replacement(df))
    #covert back to dataframe
    df_clr = pd.DataFrame(clr_np, index=df.index, columns=df.columns)
    return df_clr

def prevelance(df):
    """
    Calculate  the prevalence of variables (non zero entries)
    df --  dataframe with columns as variables and rows as observations.
    """
    prv=((df != 0).sum()/df.shape[0])*100
    prv=prv.sort_values(ascending=False)
    return prv

def calculate_corrcoef_pvalues(df,corr_stat):
    """
    Calculate Pearson (pearsonr) or spearman (spearmanr) correlation coefficient with associated p-value
    for all features in a dataframe. You will end up with a matrix of pvalues.
    Currently self correlation (i.e. feature a versus feature a) is given a value
    of 1.0. This was done to allow us to add the output to a cluster map, but also
    it should not matter since obviously there is a strong correlation between the same features.

    df -- dataframe where rows are samples (indexed) and columns are features of interest.
    corr_stat -- statistical test you wish to use in order to measure a statistical relationship
                between two variables. Must be either pearsonr or spearmanr
    """
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            if r !=c :
                tmp = df[df[r].notnull() & df[c].notnull()]
                pvalues[r][c] = corr_stat(tmp[r], tmp[c])[1]
            else:
                pvalues[r][c]=1.0000
    pvalues_matrix = pvalues[list(pvalues.reset_index()['index'])]
    return pvalues_matrix

def merge_corr_coef_pvalue_corr(df, pvalues_matrix, corr_coef=0.0, pval=0.05):
    """
    Will take your output from calculate_corrcoef_pvalues and the dataframe you used in
    generating it and return a dataframe where the first two columns are the variables
    you are calculating the association with, followed by the pvalue, corrected pvalue and
    the correlation coefficient. You can control which variables you wish to see based
    on pvalue threshold and the correlation coefficient.


    df -- dataframe where rows are samples (indexed) and columns are features of interest. Make
        sure you use the same dataframe used to generate your pvalues matrix.
    pvalues_matrix -- output from calculate_corrcoef_pvalues function
    corr_coef -- correlation coefficient you wish to use to filter your dataframe. Note:
                Will take the absolute value of the correlation coefficient
    pval -- pvalue threshold you wish to use to filter your dataframe
    """
    # get correlation matrix of your DataFrame
    corr_matrix = df.corr()

    ### Prepare the correlation df
    corr_matrix_stacked = corr_matrix.stack().reset_index()
    corr_matrix_stacked.columns = ['var1', 'var2', 'value']
    corr_matrix_stacked['absolute'] = abs(corr_matrix_stacked['value'])

    ### Prepare the pvalues calcualted above
    pvalues_matrix_stacked = pvalues_matrix.stack().reset_index()
    pvalues_matrix_stacked = pvalues_matrix_stacked.loc[ (pvalues_matrix_stacked['level_0'] != pvalues_matrix_stacked['level_1'])]
    corr_matrix_stacked = corr_matrix_stacked.rename(columns={'value':'corr_coeff'})

    # ### multiple comparisons pvalue adjustment.
    pvalues_matrix_stacked['pvalue_corr'] = multipletests(pvalues_matrix_stacked[0],
                                      alpha=0.05,
                                      method='fdr_bh',
                                      is_sorted=False,
                                      returnsorted=False)[1]

    pvalues_matrix_stacked.columns = ['var1', 'var2', 'pvalue','pvalue_corr']

    ### merge our correlation and pvalue df
    pvalues_matrix_stacked = pd.merge(pvalues_matrix_stacked, corr_matrix_stacked,  left_on=['var1','var2'], right_on = ['var1','var2'])

    # subset our dataframe based on correlation coefficient, pvalue_corr (corrected for multiple test) and also remove self correlation (feature_a vs feature_a)
    pvalues_matrix_stacked = pvalues_matrix_stacked.loc[ (pvalues_matrix_stacked['absolute'] > corr_coef) & (pvalues_matrix_stacked['var1'] != pvalues_matrix_stacked['var2']) & (pvalues_matrix_stacked['pvalue_corr']<= pval) ]
    pvalues_matrix_stacked = pvalues_matrix_stacked.drop(columns='absolute')
    return pvalues_matrix_stacked

def inverse_cov_glasso(df,filter,ncv=7,max_iter=777):
    """
    Here we use the graphical  lasso method to find the sparse  inverse covariance matrix
    with cross-validation to automatically set the alpha parameters of the l1 penalty.
    Lasso is a regression method used to induce a sparse solution to your regression
    problem by adding an L1 regularization term. Instead of estimating coefficients
    for independent variables in regression problems, graphical lasso estimates the
    precision (inverse covariance) matrix of your data. Doing so allows graphical Lasso
    to take into account the effect of other variables when examining the relationship
    between two variables. This better allows us to control the effects of other variables
    by estimating the partial correlation between two variables. This reflects
    their conditional dependence given the presence of other variables.

    df -- dataframe where rows are samples (indexed) and columns are features of interest.
    ncv -- determines the cross-validation splitting strategy. Default is 5.
    filter -- inverse covariance absolute value you wish to filter by
    max_iter -- maximum number of iterations. determines how long the algorithm will run for
                before it terminates. If the algorithm reaches the maximum number of
                iterations without converging to a satisfactory solution, it will stop and
                return the best solution found so far. Note: Increasing the maximum number of
                iterations can sometimes improve the accuracy of the model, but it can also increase
                the computation time.
    """
    # tutorials used to built funciton:
    #https://scikit-learn.org/stable/auto_examples/covariance/plot_sparse_cov.html#sphx-glr-auto-examples-covariance-plot-sparse-cov-py
    #https://towardsdatascience.com/machine-learning-in-action-in-finance-using-graphical-lasso-to-identify-trading-pairs-in-fa00d29c71a7
    edge_model = covariance.GraphicalLassoCV(cv=ncv,max_iter=max_iterr)
    df -= df.mean(axis=0)
    df /= df.std(axis=0)

    #estimate covariance
    edge_model.fit(df)

    # #the precision(inverse covariance) matrix that we want
    p = edge_model.precision_

    # add dataframe information to numpy array
    col = df.columns
    cols = pd.Series(col)
    p = pd.DataFrame(p, columns=cols, index=cols)

    # start treating the data similar to what we have been doing before to prep for
    # graphing
    networkset = p.stack().reset_index()
    networkset.columns = ['var1', 'var2','value']

    #youre threshold here can dictate what happens below
    networkset=networkset.loc[ (abs(networkset['value']) > filter) &  (networkset['var1'] != networkset['var2']) ]
    return networkset



def permutation_community_cluster(feature, G, perm, pct_present):
    """
    After creating your large network, you may be interested in identifying communities
    or modules within your community. The Louvain method for community detection is
    a method to extract communities from large networks. However, there is some randomness
    in how the algorithm begins to identify core communities and you may see the
    emergence and disappearance of particular members in a community if you continue
    to run the program. This is particularly true for weak members in a network.
    If you are interested in a particular feature and want to make sure the community
    you identify for it is robust, this permutation function  should help increase
    confidence in your results. It will randomly build a module based on your perm number
    and keep track of how many times particular features appear with one another.
    Then you could filter which features you want to keep in a community based on how
    often you see those features together. # NOTE: Currently only supporting lovian
    method; future cases could use alternative community identification methods.

    feature -- name a feature in your graph you are interested in focusing on e.g. 'Proteobacteria'.
            This function  will look for modules/communities in which this feature is a member.
    G -- this is a network graph you should have previously built e.g. G=nx.from_pandas_edgelist(networkset, 'var1', 'var2',edge_attr='value')
    perm -- number of times you wish to run the lovian method
    pct_present -- percentage of time a feature must appear in order for it to be
                included as part of this community

    """
    count=perm
    nperm = count
    final = []
    while count!=0:
        count = count-1
        partition = community_louvain.best_partition(G)
        partition_lovian = pd.DataFrame(partition.items(), columns=['Name', 'Values'])
        feat_comm=partition_lovian[partition_lovian['Name']==feature]['Values'].values[0]
        for i in list(partition_lovian[partition_lovian['Values']==feat_comm]['Name']):
            final.append(i)

    final_count  = dict(zip(list(final),[list(final).count(i) for i in list(final)]))
    final_count = pd.DataFrame.from_dict(final_count,orient='index').rename(columns={0:'appeared'})
    final_count['percent_present'] = final_count['appeared'].apply(lambda x: (x/nperm)*100)
    module_feats = final_count[final_count['percent_present'] >= pct_present]
    module_feats = list(module_feats.reset_index()['index'])
    return module_feats

def graph_degrees(graph):
    """
    Get the number of degrees for each node in a graph and sort by abundance.
    graph -- the graph you built and want to see the degrees from i.e. G=nx.from_pandas_edgelist(networkset, 'var1', 'var2',edge_attr='value')

    """
    degrees = sorted(graph.degree, key=lambda x: x[1], reverse=True)
    return degrees

def graph_betweenness(graph):
        """
        Get the betweeness centrality of a node. THis is a way of detecting the
        amount of influence a node has over the flow of information in a graph.
        graph -- the graph you built and want to see the degrees from i.e. G=nx.from_pandas_edgelist(networkset, 'var1', 'var2',edge_attr='value')

        """
        centrality = {k: v for k, v in sorted(nx.betweenness_centrality(graph).items(), key=lambda item: item[1], reverse=True)}
        return centrality

def graph_pagerank(graph,maxiter=1000):
    """
    Gte the pagerank of each node in a graph. PageRank  measures the importance
    of each node within a graph, based on the number of incoming relationships
    and the importance of the corresponding source nodes.

    graph -- the graph you built and want to see the degrees from i.e. G=nx.from_pandas_edgelist(networkset, 'var1', 'var2',edge_attr='value')
    maxiter -- maximium number of iterations in power method eigenvalue solver
    """

    pagerank = {k: v for k, v in sorted(nx.pagerank(graph,max_iter=maxiter).items(), key=lambda item: item[1], reverse=True)}
    return pagerank
