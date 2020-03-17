import os
import glob
import sys
import math
import pandas as pd
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from tqdm import tqdm

# ADD ANY OTHER IMPORTS YOU LIKE

def part1_load(folder1, folder2, n):
    dict_articles_words = {}
    corpus=[]
    articles = os.listdir(folder1)+os.listdir(folder2) 
    folders=[folder1, folder2]
    for folder in folders:
        for article in os.listdir(folder):
            words = []
            with open(folder + "/" + article) as f: # f=filename
                for line in f:
                    words += [word.lower() for word in line.split() if (word.isalpha())]
            ## find unique words in every article and their frequency in the article:
                uniqueWords, wordCount=getUnique(words)
            
            ## only select those unique words which show up more than n times:
                uniqueFrequentWords, uniqueFrequentWordCount=selectFrequentWords(uniqueWords, wordCount, n)
                corpus += [word for word in uniqueFrequentWords]
           
            ## save frequent words and their count to dictionary:
                article_plus_classname = article + '_' + folder
                for index, count in enumerate(uniqueFrequentWordCount):
                    if article_plus_classname in dict_articles_words:
                        dict_articles_words[article_plus_classname][uniqueFrequentWords[index]]=count
                    else: 
                        dict_articles_words[article_plus_classname]={}
                        dict_articles_words[article_plus_classname][uniqueFrequentWords[index]]=count
                
    # extract class name from the file extention the files were previously given:
    k = [k.partition("_")[2] for k,v in dict_articles_words.items()]
    
    #fill out NaN cells with 0's:
    df = pd.DataFrame(dict_articles_words).fillna(0) 
    
    #transpose the dateframe so that x-axis becomes y-axis and vice versa: 
    df_transposed = df.T
    
    #add a column 'class_name' to the dataframe:  
    df_transposed.insert(0,'class_name', k, True)
   
    
    return df_transposed
    
def getUnique(x):
    y, f = np.unique(x, return_counts=True)
    return y, f

def selectFrequentWords(words, counts, n):
    more_than_n_times = []
    remaining_counts = []
    for index, count in enumerate(counts):
        if count > n:
                more_than_n_times.append(words[index])
                remaining_counts.append(count)
    return more_than_n_times, remaining_counts


def part2_vis(df, m):
    assert isinstance(df, pd.DataFrame)
    
    #summerize and sort all columns (after 'classname', )
    df_sum = df.sum()[2:].sort_values(ascending = False)
    #take out m most frequent words:
    df_top = df_sum[:m]
    #filter input dataframe on top m frequent words: 
    df_filtered = df.filter(df_top.index, axis=1)
    #insert class name column again:
    df_filtered.insert(0, 'class_name', df['class_name'])
    #group and summarize columns by class name:
    df_grouped = df_filtered.groupby(['class_name']).sum().sort_values(by=['class_name'])

    return df_grouped.T.plot(kind="bar")

def part3_tfidf(df):
    # DO NOT CHANGE
    assert isinstance(df, pd.DataFrame)

    noOfDocs = len(df.index)
    #remove class column in order to loop through word columns:
    df_without_class=df.iloc[:, 1:] 
    #look at the number of cells in each word column that don't contain zeros and save that number as document frequency
    docFreqList=[]
    for (columnName, columnData) in df_without_class.iteritems():
        docFreq = len([int(freq) for freq in columnData if int(freq) > 0])
        docFreqList.append(docFreq)
   
    #make a copy of the input dataframe's indices and data:
    df2=df.copy(deep=True)
    #make calucations on items in docFreqList and save in tdidfList: 
    tfidfList=[]
    for freq in docFreqList:
        if(freq==0):
            tfidfList.append(1)
        else:
            tfidfList.append(np.log(noOfDocs/freq))
    
    #populate new dataframe with values from tfidfList:
    df2.iloc[:, 1:]=df.iloc[:, 1:]*np.array(tfidfList)
    df2.iloc[:, 0]=df.iloc[:, 0]

    return df2


# ADD WHATEVER YOU NEED HERE, INCLUDING BONUS CODE.

#part1_load('grain/', 'crude/', 1)