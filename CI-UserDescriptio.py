# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 10:20:26 2018

@author: mdavala
"""

from twython import Twython, TwythonError, TwythonRateLimitError
import pandas as pd
import warnings
import time

warnings.filterwarnings(action='once')
client_args = {
    'verify': False
}


t = Twython(app_key='', 
            app_secret='', 
            oauth_token='', 
            oauth_token_secret='',
            client_args = client_args)

t.verify_credentials()


#df = pd.read_excel('Psychology1.xlsx', None)
df = pd.read_excel('Medical7.xlsx',None)

#writer = pd.ExcelWriter('PsySheet1_UserDescription.xlsx')
writer = pd.ExcelWriter('MedSheet7UserDescription.xlsx')

for i in df.keys():
    print (i)
    tmp = df[i]
    tmp = tmp.dropna(subset = ['twitter_user_page'], how ='all')
    tmp = tmp.dropna(subset = ['tweet_post_url'], how ='all')
    output = pd.DataFrame(columns = ['Altid','twitter_user_page', 'user description', 
                                'followers_count', 'following_count',
                                'timezone', 'post_retweets', 'post_likes'])
    colu = list(output.columns)
    for j in range(len(tmp['twitter_user_page'])):
        user = tmp['twitter_user_page'].iloc[j][20:]
        url = tmp['tweet_post_url'].iloc[j]
        post_id = url[url.rfind('/')+1:]
        altid = tmp['altid'].iloc[j]
        user_link = tmp['twitter_user_page'].iloc[j]
        print (j)
        try:
            post_data = t.lookup_status(id = post_id)
            post_data = post_data[0]         
            description = post_data['user']['description']
            followers_count = post_data['user']['followers_count']
            following_count = post_data['user']['friends_count']
            post_retweets = post_data['retweet_count'] 
            time_zone = 'NA'
            post_likes = post_data['favorite_count']
            if 'retweeted_status' in post_data.keys():
                time_zone = post_data['retweeted_status']['user']['time_zone']
                post_likes = post_data['retweeted_status']['favorite_count']
            
            res = [altid, user_link, description, followers_count, following_count,
                   time_zone, post_retweets, post_likes]
            output = output.append(pd.DataFrame(columns =colu, data=[res]), 
                                   ignore_index = True)
        
        except TwythonRateLimitError as error:
            print ("[Exception Raised] Rate limit exceeded")
            reset = int(t.get_lastfunction_header('x-rate-limit-reset'))
            wait = max(reset - time.time(), 0) + 10
            time.sleep(wait)
        except IndexError as ie:
            print ("Index Error: User No longer available")
            print (ie)
            output = output.append(pd.DataFrame(columns = colu, data = [[altid, user_link, 'NA', 'NA', 'NA', 'NA', 'NA', 'NA']]), 
                                   ignore_index=True)
        except Exception as e:
            print ("Non rate-limit exception encountered. Sleeping for 15 min before retrying")
            print (e)
            #time.sleep(60*15)
            output = output.append(pd.DataFrame(columns = colu, data = [[altid, user_link, 'NA', 'NA', 'NA', 'NA', 'NA', 'NA']]), 
                                   ignore_index=True)
            
        
    output.to_excel(writer, i, index=False)
writer.save()
