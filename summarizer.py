# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 01:22:22 2018

@author: mdavala
"""

import pandas as pd
import numpy as np

articleCount = pd.read_excel('Articles/Medical_#1to#20.xlsx', sheet_name='Sheet1')
medLabel_data = pd.read_excel('Medical_labelled.xlsx', None)
articleCount = articleCount.dropna(subset=['AltmetricID'], how='all')

writer = pd.ExcelWriter('Medical Summary.xlsx')
sheetname = 'Medical Summary'

summary = pd.DataFrame(columns = ['AltmetricID', 'Tweet Count', 'Total English', 
                                 'academic tweeters', 'non-academic tweeters',
                                 'individual tweeters', 'organization tweeters', 
                                 'total followers', 'academic followers',
                                 'non-academic followers', 'individual followers',
                                 'organization followers', 'article retweets',
                                 'article likes'])
summary['AltmetricID'] = articleCount['AltmetricID']
summary['Tweet Count'] = articleCount['Tweets']

for i in medLabel_data.keys():
    print (i)
    temp = medLabel_data[i]
    for j in temp['Altid'].unique():
        print (j)
        total_english = len(temp[(temp['Altid'] == j)]['acad labels'])
        a = temp[(temp['Altid'] == j)]['acad labels']
        b = temp[(temp['Altid'] == j)]['individual labels']
        
        #Labeling
        acad_count = len(a[a==1])
        nonAcad_count = len(a[a==0])
        indiv_count = len(b[b==0])
        org_count = len(b[b==1])
        
        #Followers
        total_followers = sum(temp[(temp['Altid'] == j)]['followers_count'])
        acad_followers = sum(temp[(temp['Altid'] == j) & (temp['acad labels']==1)]['followers_count'])
        nonAcad_followers = sum(temp[(temp['Altid'] == j)& (temp['acad labels']==0)]['followers_count'])
        indiv_followers = sum(temp[(temp['Altid'] == j) & (temp['individual labels']==0)]['followers_count'])
        org_followers = sum(temp[(temp['Altid'] == j) & (temp['individual labels']==1)]['followers_count'])
        indiv_acad_ppl = len(temp[(temp['Altid'] == j) & (temp['individual labels']==0) & (temp['acad labels']==1)])
        indiv_nonAcad_ppl = len(temp[(temp['Altid'] == j) & (temp['individual labels']==0) & (temp['acad labels']==0)])
        org_acad_ppl = len(temp[(temp['Altid'] == j) & (temp['individual labels']==1) & (temp['acad labels']==1)])
        org_nonAcad_ppl = len(temp[(temp['Altid'] == j) & (temp['individual labels']==1) & (temp['acad labels']==0)])
        article_retweets = sum(temp[(temp['Altid'] == j)]['post_retweets'].unique())
        article_likes = sum(temp[(temp['Altid'] == j)]['post_likes'].unique())
        
        
        summary.loc[summary['AltmetricID'] == j, 'Total English'] = total_english
        summary.loc[summary['AltmetricID'] == j, 'academic tweeters'] = acad_count
        summary.loc[summary['AltmetricID'] == j, 'non-academic tweeters'] = nonAcad_count
        summary.loc[summary['AltmetricID'] == j, 'individual tweeters'] = indiv_count
        summary.loc[summary['AltmetricID'] == j, 'organization tweeters'] = org_count
        summary.loc[summary['AltmetricID'] == j, 'total followers'] = total_followers
        summary.loc[summary['AltmetricID'] == j, 'academic followers'] = acad_followers
        summary.loc[summary['AltmetricID'] == j, 'non-academic followers'] = nonAcad_followers
        summary.loc[summary['AltmetricID'] == j, 'individual followers'] = indiv_followers
        summary.loc[summary['AltmetricID'] == j, 'organization followers'] = org_followers
        summary.loc[summary['AltmetricID'] == j, 'article retweets'] = article_retweets
        summary.loc[summary['AltmetricID'] == j, 'article likes'] = article_likes
        summary.loc[summary['AltmetricID'] == j, 'individual academic'] = indiv_acad_ppl
        summary.loc[summary['AltmetricID'] == j, 'individual non-acad'] = indiv_nonAcad_ppl
        summary.loc[summary['AltmetricID'] == j, 'organization academic'] = org_acad_ppl
        summary.loc[summary['AltmetricID'] == j, 'organization non-acad'] = org_nonAcad_ppl
summary.to_excel(writer, sheet_name=sheetname, index= False)
writer.save()
        
        
        
        
        
        
        

