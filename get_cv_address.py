import os
import pandas as pd

p_root = 'D:\ei\Sentiment_Analysis_Imdb-master\p'
n_root = 'D:\ei\Sentiment_Analysis_Imdb-master/n'


print(os.listdir(n_root))
address_li = []
label_li = []

for it in os.listdir(p_root):
    for item in os.listdir(p_root+'/'+it):
        name = item
        if '.jpg' in name:
            address_li.append(p_root+'/'+it+'/'+name)
            label_li.append(1)

for it in os.listdir(n_root):
    for item in os.listdir(n_root+'/'+it):
        name = item
        if '.jpg' in name:
            address_li.append(n_root+'/'+it+'/'+name)
            label_li.append(0)

df = pd.DataFrame({'labels': label_li,'sentences': address_li})

df.to_csv('D:/ei/Sentiment_Analysis_Imdb-master/cv_address.csv', index=False, encoding="utf-8")
