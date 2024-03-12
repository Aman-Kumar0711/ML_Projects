#!/usr/bin/env python
# coding: utf-8

# In[108]:


import pandas as pd


# In[109]:


teams=pd.read_csv("teams.csv")


# In[110]:


teams


# In[111]:


team=teams[["year","athletes","age","prev_medals","medals"]]


# In[112]:


team


# In[113]:


team.corr()["medals"] #iss function se hum medals column ka correlation find kar rahe hain baaki columns ke saath iss se hum yeh pata kar paayenge ki kis column ke through predict karna zyada sahi rahega 


# In[114]:


#we can use prev_medals column to predict values of medals column since iska sabse zyada correlation hai
teams=teams[["team","country","year","athletes","age","prev_medals","medals"]]
#ab mujhe team and country columns ki bhi zarurat padd sakti hai to unko bhi include kar liya


# In[115]:


#chalo ab graphical form mei check kar lete hain relation iske liye we will import a graphing library of python and that is seaborn


# In[116]:


import seaborn as sns


# In[117]:


sns.lmplot(x="athletes",y="medals",data=teams,fit_reg=True,ci=None) #agar ci =None nahin karenge to seaborn humko ek confidence interval line de dega graph mei and fit_reg true karke ek regression line aa jaayegi poore data mei


# In[118]:


#iss upar waale graph ko dekh ke yeh pata chalta hai ki jis country mei number of athletes zyada honge usme medals zyada hain


# In[119]:


sns.lmplot(x="age",y="medals",data=teams,fit_reg=True,ci=None)


# In[120]:


#okay matlab age ka zyada koi khaas sambandh nahin hai no. of medals won se...so we'll not use this for prediction


# In[121]:


teams.plot.hist(y="medals")


# In[122]:


#ab thodi data cleaning karenge kyun ki humaare data mei bahut si missing values thi'n ab unhi ko remove karenge


# In[123]:


teams[teams.isnull().any(axis=1)]


# In[124]:


teams=teams.dropna()


# In[125]:


teams


# In[126]:


#ab iss data ko 2 hisso mei baant lenge kyun ki agar mujhe 2012 se baad ki years mei number of medals predict karne hain to main 2012 se pehle ke data ko use karunga na as training dataset and 2012 se baad waale ko main test dataset mei daalunga


# In[127]:


train=teams[teams["year"]<2012].copy()
test=teams[teams["year"]>=2012].copy()


# In[128]:


train.shape


# In[129]:


test.shape


# In[130]:


#generally 80:20 train/test split karte hain data ko 


# In[131]:


#ab train karenge model ko, chalo linear regression use karte hain


# In[132]:


from sklearn.linear_model import LinearRegression
reg=LinearRegression()


# In[133]:


predictors=["athletes","prev_medals"] #sabse zyada correlation inhi 2 columns ka aa raha tha medals ke saath to inhi ko as predictors use kar rahe hain
target="medals"


# In[134]:


reg.fit(train[predictors],train["medals"]) #.fit() se fit karenge linear regression model ko apne


# In[135]:


predictions=reg.predict(test[predictors]) #test data ke predictors isliye use kare hain kyun ki we don't want our model to know about thr exact answers hence woh khud hi predict kare


# In[136]:


predictions


# In[137]:


#yeh to whole number mei nahin hain number of medals and kuchh to negative bhi hain ...chalo sahi karte hain isko


# In[138]:


#pehle to isko ek column ke format mei assign kar dete hain apne test data mei


# In[139]:


test["predictions"]=predictions


# In[140]:


test


# In[141]:


#chalo jahaan prediction <0 hai usko 0 se replace kar denge


# In[142]:


test.loc[test["predictions"]<0,"predictions"]=0


# In[143]:


test["predictions"]=test["predictions"].round()


# In[144]:


test


# In[145]:


#chalo ab mean abs error find karte hain predictions and actual medals mei


# In[146]:


from sklearn.metrics import mean_absolute_error
error=mean_absolute_error(test["medals"],test["predictions"])


# In[147]:


error


# In[148]:


#let's see kaisa error hai yeh


# In[149]:


teams.describe()["medals"]


# In[150]:


#humaara error agar standard deviation ke andar hai to sahi maani jaati hai prediction...so iss case mei std 33 hai and humaara error keval 3 hai so we are good to go


# In[151]:


#chalo ab team by team check karte hain predictions


# In[152]:


test[test["team"]=="USA"]


# In[153]:


test[test["team"]=="IND"]


# In[154]:


#ab upar India ne 2016 mei 2 medal jeete hain per humaari prediction 12 bata rahi hai to mean abs error se hum different countries ke liye different data paayenge in percentage terms...USA ke case mei 248 medals thhe actually and humaara ans 285 tha to woh to koi dikkat ki baat nahin hai but 2 ki jagah 12 ho jaana to thodi samasya ki baat hai


# In[155]:


#chalo ab country basis per check kar lete hain kitna difference hai actual and predicted medals mei
errors=(test["medals"]-test["predictions"]).abs()


# In[156]:


errors


# In[157]:


#ab iss data ko group kar dete hain with team then mean find kar lenge uss se pata chal jaayega on an average har country mei kitna medal difference hai


# In[158]:


error_by_team=errors.groupby(test["team"]).mean()


# In[159]:


error_by_team


# In[163]:


#let's check how many medals each country earned on average
medals_by_team=test["medals"].groupby(test["team"]).mean()


# 

# In[165]:


#let's find the ratio between error_by_team and medals_by_team
error_ratio=error_by_team/medals_by_team


# In[166]:


error_ratio


# In[167]:


#upar to bahut si missing values hain(kyun ki bahut si countries ne ek bhi medal nahin jeeta hai and error bhi zero hai to we are basically dividing by 0 by 0  in ratio)


# In[168]:


#let's take only those values which are not missing
error_ratio[~pd.isnull(error_ratio)]


# In[169]:


#ab inf waali values ko bhi hatana hai, inf tab aata hai ab numerator non zero ho and denominator 0 ho


# In[170]:


import numpy as np
error_ratio=error_ratio[np.isfinite(error_ratio)]


# In[171]:


error_ratio


# In[172]:


#let's make histogram of this
error_ratio.plot.hist()


# In[173]:


#ab dekho kuchh countries mei error ratio 12% aa rha hai kuchh bhi 36% and so on ...yeh to theek hai but jinme twice as high aa rha hai woh dikkat hai to kitna theek hai kitna nahin model yeh hum apni demands ke hisaab se decide kar sakte hain and hum isko aur accurate bana sakte hain


# In[174]:


#haan but iss se hum yeh predict kar sakte hain ki kaunsi country ko medal milne ke chances zyada hain, unme error ratio bahut kam hoga


# In[175]:


error_ratio.sort_values()


# In[176]:


#if we want to improve accuracy and performance of this model then we can do these things:
#add more predictors
#try different ML models like random forest
#aur advanced data use kar sakte hain jahaan aur zyada info ho jaise iss waale mei har country ke athletes ka individual data hota to aur achhe se predict kar paate
#we can reshape out data in case of non linear correlation by using different mathematical transformations
#measure error more proficiently


# In[ ]:




