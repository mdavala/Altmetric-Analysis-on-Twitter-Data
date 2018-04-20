# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 15:24:08 2018

@author: mdavala
"""

import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import math


output = pd.DataFrame(columns = ['altid', 'datetime', 'twitter_user_page',
                                'twitter_user_name', 'tweet_post_url'])

#df1 = pd.read_excel("code_+_data/Psy_#1to#20.xlsx")
df1 = pd.read_excel("code_+_data/Psy_#1to#20.xlsx")
df1 = df1.dropna(subset=['AltmetricID'])
df1['AltmetricID'] = df1.AltmetricID.astype(np.int64)

#Store all the altmetric ID in array
altIDs = df1.AltmetricID.unique()

######################### USER DEFINED FUNCTIONS ############################
def scrape_pages(id):
    url = 'https://explorer.altmetric.com/details/%s/twitter/page:1'%str(id)
    res = requests.get(url)
    soup = BeautifulSoup(res.content,'html.parser')
    joinlist = []
    try:
        tag_panel = soup.find_all('div',{'class':'tab-panel'})[0].text
        if tag_panel.find('Twitter') > 0:         
            try:
                page_num = soup.find_all('div',{'class':'post_pagination top'})[0].find_all('a')
                tweet_pages = []
                for ele in page_num:
                    tweet_page = ele.get('href')
                    atpos = tweet_page.find(':')
                    tweet_pages.append(int(tweet_page[atpos+1 : ]))
                page_range = range(1,max(tweet_pages)+1)
            except:
                page_range = "1"
            page_list = []
            for i in page_range:
                page = ''.join([url[:-1],str(i)])
                page_list.append(page)
            for page in page_list:
                res2 = requests.get(page)
                soup2 = BeautifulSoup(res2.content,'html.parser') 
                twitter = soup2.find_all('article',{'class':'post twitter'})
                for item in twitter:
                    altid = page.split('/')[4]
                    user = item.contents[1]
                    li = item.contents[4]
                    datetime = li.get('datetime')
                    twitter_user_page = user.get('href')
                    twitter_user_name = user.find_all('div',{'class':'name'})[0].text
                    tweet_post_url = li.find_all('a')[0].get('href')
                    
                    export = [altid, datetime, twitter_user_page,
                              twitter_user_name,tweet_post_url]
                    joinlist.append(export)
    except:
        joinlist = [[str(id), 'NA', 'NA', 'NA', 'NA']]
    return joinlist

colu = list(output.columns)
for i in range(len(altIDs)):
    res = scrape_pages(altIDs[i])
    output = output.append(pd.DataFrame(columns= colu, data = res), ignore_index = True)

chunks = np.array_split(output, math.ceil(len(output)/32000))    
no_of_files = math.ceil(len(chunks)/10)
for i in range(no_of_files):
    writer = pd.ExcelWriter('Psychology'+str(i+1)+'.xlsx')
    for x in range(len(chunks)):
        if x==10:
            chunks = chunks[10:]
            break
        sheetname = 'sheetname'+str(x+1)
        chunks[x].to_excel(writer, sheetname)
    writer.save()



























        
        