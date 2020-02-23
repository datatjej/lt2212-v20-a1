import os
import glob
import sys
import pandas as pd
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
#from collections import Counter
#import nltk

#nltk.FreqDist()

# ADD ANY OTHER IMPORTS YOU LIKE

#C502

# DO NOT CHANGE THE SIGNATURES OF ANY DEFINED FUNCTIONS.
# YOU CAN ADD "HELPER" FUNCTIONS IF YOU LIKE.

#print(os.getcwd('/grain'))
#print(os.listdir(os.getcwd('/grain')))
#print(glob.glob('/grain')+"/*")

#Either a list of lists, or a dictionary

def part1_load(folder1, folder2, n=1):
    dict_articles_words = {}
    corpus=[]
    articles = os.listdir(folder1)+os.listdir(folder2) #filenames inside crude
    folders=[folder1, folder2]
    for folder in folders:
        for article in articles:
            words = []
            with open(folder + "/" + article) as f: # f=filename
                for line in f:
                    words += [word.lower() for word in line.split() if (word.isalpha())]
            ## find unique words and their frequency
                uniqueWords, wordCount=getUnique(words)

            ## only select those unique words which show up more than n times
                uniqueFrequentWords, uniqueFrequentWordCount=selectFrequentWords(uniqueWords, wordCount, n)
                corpus += [word for word in uniqueFrequentWords]
                #print("length of unique words=", len(uniqueFrequentWords))
            ## saven frequent words and their count to dictionary:
                #temp_dict={}
                article_plus_classname = article + '_' + folder
                for index, count in enumerate(uniqueFrequentWordCount):
                    if article_plus_classname in dict_articles_words:
                        dict_articles_words[article_plus_classname][uniqueFrequentWords[index]]=count
                    else: 
                        dict_articles_words[article_plus_classname]={}
                        dict_articles_words[article_plus_classname][uniqueFrequentWords[index]]=count

                
                #temp_dict.update({uniqueFrequentWords[index]: count})
                #for index, count in enumerate(uniqueFrequentWordCount):
                #dict_articles_words.update({article + '_' + folder1: temp_dict})
               
    #print("shape dict", len(dict_articles_words))
    #print("dictionary", dict_articles_words)
    #print(corpus)
    
    k = [k.partition("_")[2] for k,v in dict_articles_words.items()]
    
    df = pd.DataFrame(dict_articles_words).fillna(0) 
    df_transposed = df.T
    df_transposed.insert(0,'class', k, True)
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
    # DO NOT CHANGE
    assert isinstance(df, pd.DataFrame)
    df_subset=df[df['class']=='grain']
    df_subset=df_subset.sum().to_frame()
    df_subset_grain=df_subset[1:]
    df_subset_grain.columns = ['words']
    
    df_subset=df[df['class']=='crude']
    df_subset=df_subset.sum().to_frame()
    df_subset_crude=df_subset[1:]
    df_subset_crude.columns = ['words']
    
    df_total=df_subset_grain + df_subset_crude
    df_total=df_total.sort_values(by=['words'], ascending = False)
    l_crude=[]
    l_grain=[]
    for word in df_total[:m].index:
        l_crude.append(df_subset_crude.loc[word, 'words'])
        l_grain.append(df_subset_grain.loc[word, 'words'])
    
    plt.figure(figsize=(12, 6))
    labels=[word for word in df_total[:m].index]

    x = np.arange(len(labels))  # the label locations

    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, l_crude, width, label='crude')
    rects2 = ax.bar(x + width/2, l_grain, width, label='grain')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    plt.legend()
    


def part3_tfidf(df):
    # DO NOT CHANGE
    assert isinstance(df, pd.DataFrame)

    # CHANGE WHAT YOU WANT HERE
    return df #DUMMY RETURN

# ADD WHATEVER YOU NEED HERE, INCLUDING BONUS CODE.

part1_load('grain/', 'crude/', 1)