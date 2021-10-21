#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')



#Scarrter plot and Pearson correlation (Jingyi Gao)

d = pd.DataFrame.from_dict({'X':[590,520,600,580,590,560,570,590,580,600,610,690,600,570,380,160,190,290,230,130,150,210,430,580,500],'Y':[5,5.1,4.9,5,4.5,4.6,4.8,5,4.7,4.5,4.2,4.9,5.9,5.7,5.5,6.1,6.5,7,6.7,7,6.5,6.7,6.6,6.4,6.7]})
print(d)
plt.scatter(d.loc[:,'X'],d.loc[:,'Y'])
plt.title("Retail sales vs Unemployment rate")
plt.xlabel("Retail sales")
plt.ylabel("Unemployment rate")
plt.show()
print("The Pearson correlation value r is ",d['X'].corr(d['Y']))




# Scatter plot and calculate Pearson correlation(SEN WU) 
d = pd.DataFrame.from_dict({'X' : [21.30,20.125,21.1,20.7,21.025,19.975,17.625,15.725,13.925,13.275],  'Y' :[5.0,5.4,5.7,6.2,6.5,5.9,5.9,5.6,4.6,5.4]})
print(d)
plt.scatter(d.loc[:,'X'],d.loc[:,'Y'])
plt.title("Affordable rental vs Unemployment rate")
plt.xlabel("Affordable rental")
plt.ylabel("Unemployment rate")
plt.show()
print("Pearson r is ",d['X'].corr(d['Y']))




#Yiqun Yang
rate_employment_growth = [2.8,1.0,1.2,0.9,2.4,2.7,4.0,2.8,3.4,1.2,-1.0]
rate_rgsp_growth = [3.0,2.4,0.9,2.0,2.8,3.5,3.8,3.4,3.1,-0.5,-2.0]
Figure, axis = plt.subplots()
axis.set_title("Scatterplot for rate of RGSP growth and rate of Employment growth")
axis.set_xlabel("tha rate of employment growth")
axis.set_ylabel("rate of RGSP growth")
axis.scatter(rate_employment_growth, rate_rgsp_growth)
num=0
count=0
for i in rate_employment_growth:
    num+=i
    count+=i
x_mean = num/count
num=0
count=0
for i in rate_rgsp_growth:
    num+=i
    count+=i
y_mean = num/count
x_time_y = 0
denominator_x = 0
denominator_y = 0
numerator=0
r_value=0
for i in range(len(rate_employment_growth)):
    numerator=(rate_employment_growth[i]-x_mean)*(rate_rgsp_growth[i]-y_mean)
    denominator_x+=(rate_employment_growth[i]-x_mean)**2
    denominator_y+=(rate_rgsp_growth[i]-y_mean)**2
    x_time_y+=numerator
r_value=x_time_y/math.sqrt(denominator_x*denominator_y)
print("The pearson correlation value is", r_value)




#Calculate entropy information(SEN WU)
def my_entropy(probs):
    return -probs.dot(np.log2(probs))

d =pd.DataFrame.from_dict({'X' : [6,6,6,6,6,6,5,4,4,4], 'Y' :[1,2,2,3,3,2,2,2,1,2]})

d['X']
print("H(X)",my_entropy(d['X'].value_counts(normalize=True, sort=False)))
print("H(Y)",my_entropy(d['Y'].value_counts(normalize=True, sort=False)))

#Calculate Mutual information(SEN WU)
def mutual_info(df):
    
    Hx = my_entropy(df.iloc[:,0].value_counts(normalize=True, sort=False))
    Hy = my_entropy(df.iloc[:,1].value_counts(normalize=True, sort=False))
      
    counts = d.groupby(["X","Y"]).size()
    probs = counts/ counts.values.sum()
    H_xy = my_entropy(probs)

    # Mutual Information
    I_xy = Hx + Hy - H_xy
    MI = I_xy
    NMI = I_xy/min(Hx,Hy) #I_xy/np.sqrt(H_x*H_y)
    
    
    return {'Hx':Hx,'Hy':Hy,'MI':MI,'NMI':NMI} 

d =pd.DataFrame.from_dict({'X' : [6,6,6,6,6,6,5,4,4,4], 'Y' :[1,2,2,3,3,2,2,2,1,2]})
mutual_info(d)



# In[ ]:




