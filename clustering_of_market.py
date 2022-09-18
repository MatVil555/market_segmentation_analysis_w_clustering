
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set()

from sklearn.cluster import KMeans
from sklearn import preprocessing

data=pd.read_csv('Market_segmentation.csv')
#data=pd.read_csv('Countries_with_langauges.csv')
dt=data.copy()
print(data.head())

plt.scatter(data['Satisfaction'],data['Loyalty'])
plt.xlabel('Sat')
plt.ylabel('Loy')
plt.show()

#I need to standardize the features 
#because Loyaaty adn Satisfaction have different scale, 
#and so the bigger one, (in this case Satisfaction) 
#would be (almost) the only important criteria for clustering

x_scaled=preprocessing.scale(data)

#back to Pandas
df = pd.DataFrame(x_scaled, columns = ['Satisfaction','Loyalty'])

print(x_scaled)


# define features
df=df.iloc[:,0:2]


#define algoritm for 3 clusters....
kmeans=KMeans(3)
n_cluster=kmeans.fit_predict(df)
df['Clusters']=n_cluster
print(df)

#plot of bad clustering
plt.scatter(df['Satisfaction'],df['Loyalty'],c=df['Clusters'],cmap='rainbow')
plt.show()


#find the proper number of clusters with elbow method
elb=[]

max_try=15 #max number of clusters we will try for
for i in range(1,max_try):
    kmeans=KMeans(i)
    kmeans.fit(df)
    
    elb.append(kmeans.inertia_)
    

plt.plot(range(1,max_try),elb)
plt.xlabel('Number of clusters')
plt.ylabel('WCSS (Thithing Clusters Sum Squared)')
plt.show()
# i can do 4 or 5 ebcause it minimizes WCSS and the number of clusters
#most important, I have to plot the non-standardized values.

#define algoritm for 3 clusters....
kmeans=KMeans(4)
n_cluster=kmeans.fit_predict(df)
df['Clusters']=n_cluster
print(df)

plt.scatter(data['Satisfaction'],data['Loyalty'],c=df['Clusters'],cmap='rainbow')
plt.xlabel('Satisfaction')
plt.ylabel('Loyalty')
plt.show()

#


