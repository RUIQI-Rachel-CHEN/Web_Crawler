#!/usr/bin/env python
# coding: utf-8

# In[141]:


from selenium import webdriver
import time
import pandas as pd


# In[142]:


# Start the Driver
driver = webdriver.Chrome(executable_path = r"./chromedriver")


# In[143]:


news_contents=[]
news_titles=[]
news_times=[]
news_url=[]
Symbols=["AAPL","FB","GOOG"]

for i in Symbols:
    pageUrl ='https://finance.yahoo.com/quote/'+ i + '/press-releases?p='+ i  
    driver.get(pageUrl)
    for i in range(0,3):
        driver.execute_script("window.scrollBy(0,5000)")
        time.sleep(3)
    for i in driver.find_elements_by_css_selector("a.Fw\\(b\\).Fz\\(18px\\).Lh\\(23px\\).Fz\\(17px\\)--sm1024.Lh\\(19px\\)--sm1024.mega-item-header-link.Td\\(n\\).not-isInStreamVideoEnabled"):
        news_url.append(i.get_attribute("href"))
    for i in news_url:
        #content_url
        content_url=i
        driver.get(content_url)
        time.sleep(1)
        #title
        title=driver.find_elements_by_css_selector("h1[itemprop='headline']")[0].text
        #time
        time0=driver.find_elements_by_css_selector("time[itemprop='datePublished']")[0].text

        #content
        content_p1=driver.find_elements_by_css_selector("article[itemprop='articleBody']")[0]
        content_ps=content_p1.find_elements_by_css_selector("p")
        content='\n'.join([p.text for p in content_ps])

        news_contents.append(content)
        news_titles.append(title)
        news_times.append(time0)


# In[146]:


df=pd.DataFrame({'News Title':news_titles,'Release Date':news_times,'News Ccontent':news_contents})
df


# In[89]:


df.to_csv('news68.csv')


# In[ ]:




