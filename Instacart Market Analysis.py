#!/usr/bin/env python
# coding: utf-8

# # Instacart Market Analysis

# ## Project Introduction

# #### What is Instacart?
# 
# Instacart operates a grocery delivery and pick-up service in the United States and Canada. It operates in over 5,500 cities and its customers place orders through both website and mobile apps. Instacart partners with over 350 retailers that have more than 25K grocery stores. After the customer places their orders through the app, personal shoppers review your order and do the in-store shopping and delivery for you.
# 
# Instacart uses the order data from the customers to be able to understand customer purchase patterns. They use this data to learn about the products a user will buy again, or add to their cart next during a particular shopping session.
# 
# #### What are we trying to solve?
# 
# In this project, we will try and understand some patterns in customer behavior. Some of the questions we deal with in this project are :
# 
# What time and day of the week are the customers placing the orders?
# What are the products which drive the orders i.e. products which make or break the order?
# What are the products that are being re-ordered?
# What percentage of products being ordered are perishables / non-perishables?
# What are the departments / aisles that have the most number of orders? and re-orders?
# Does the re-order ratio of products change by time or day of the week?
# Is there any relation between the order in which the products are added to the cart with the re-order of that product?
# How frequently does the customer place the order on the Instacart app?
# How many products does a customer generally buy in each order?
# Once we get a fair idea of the purchase pattern of the customers, we also build 3 models which will help us predict the customer behavior in the future. This will help Instacart while partnering with retailers. It also helps to get the products which need to be stocked up, cross sell products etc.

# In[ ]:


Loading the required packages:


# In[105]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
from itertools import combinations, groupby
from collections import Counter

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


Importing all datasets:


# In[82]:


#import all datasets
order_products_train = pd.read_csv("C:/Users/arunp/Desktop/UC/ACADEMICS/PROJECTS/INSTACART MARKET/order_products__prior.csv")
order_products_test = pd.read_csv("C:/Users/arunp/Desktop/UC/ACADEMICS/PROJECTS/INSTACART MARKET/order_products__train.csv")
orders = pd.read_csv("C:/Users/arunp/Desktop/UC/ACADEMICS/PROJECTS/INSTACART MARKET/orders.csv")
products = pd.read_csv("C:/Users/arunp/Desktop/UC/ACADEMICS/PROJECTS/INSTACART MARKET/products.csv")
aisles = pd.read_csv("C:/Users/arunp/Desktop/UC/ACADEMICS/PROJECTS/INSTACART MARKET/aisles.csv")
departments = pd.read_csv("C:/Users/arunp/Desktop/UC/ACADEMICS/PROJECTS/INSTACART MARKET/departments.csv")


# In[ ]:


#### DATA CLEANING AND EXPLORATION


# In[ ]:


Before we dive deep into the exploratory analysis, let us know a little more about the datasets given.


# In[83]:


def details(variable):
    print ("Rows     : " ,variable.shape[0])
    print ("Columns  : " ,variable.shape[1])
    print ("\nFeatures : \n" ,variable.columns.tolist())
    print ("\nMissing values :  ", variable.isnull().sum().values.sum())
    print ("\nUnique values :  \n",variable.nunique())
    
details(order_products_train)


# In[84]:


details(order_products_test)


# In[85]:


details(orders)


# In[ ]:



There are 206,209 missing values in the column 'days_since_prior_order'. This means that these orders are the first orders for the 206,209 unique customers.
We will impute these missing values with a random value '999' to distinguish them.


# In[86]:


orders['days_since_prior_order'].fillna(999,inplace = True)


# In[50]:


details(products)


# In[39]:


details(aisles)


# In[40]:


details(departments)


# We will now merge all the datasets together to explore more about the data.

# In[51]:


cart_merged = pd.merge(order_products_train,orders, on = 'order_id')
cart_merged = pd.merge(cart_merged,products, on = 'product_id')
cart_merged = pd.merge(cart_merged,aisles, on = 'aisle_id')
cart_merged = pd.merge(cart_merged,departments, on = 'department_id')


# In[53]:


details(cart_merged)


# In[ ]:


#### Count of orders by customers


# In[ ]:


Now let us validate the claim that 4 to 100 orders of a customer are given.


# In[59]:


#order count
cnt_ord = orders.groupby("user_id")["order_number"].aggregate(np.max).reset_index()
cnt_ord = cnt_ord.order_number.value_counts()

plt.figure(figsize=(15,10))
sns.barplot(cnt_ord.index, cnt_ord.values, alpha=0.8, color='gray')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Maximum order number', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()


# We can observe that there are no orders less than 4 and the maximum order by a customer is capped at 100 as given in the data page

# #### Orders by Day of Week

# Now let us see how the ordering habit changes with day of week.

# In[66]:


#ordersperday
plt.figure(figsize=(12,8))
sns.countplot(x="order_dow", data=orders, color='gray')
plt.ylabel('Number of Orders', fontsize=12)
plt.xlabel('Day of week', fontsize=12)
plt.title("Frequency of order by day of week", fontsize=15)
plt.show()


# Seems like 0 and 1 being Saturday and Sunday has the maximum number of orders and order numbers are the lowest during Wednesday.

# In[ ]:


#### Orders by Hour of Day


# In[ ]:


Now , let us see how the distribution of orders is changing with respect to the time of day.


# In[69]:


#ordersperhouroftheday
plt.figure(figsize=(12,8))
sns.countplot(x="order_hour_of_day", data=orders, color='gray')
plt.ylabel('Number of Orders', fontsize=12)
plt.xlabel('Day of week', fontsize=12)
plt.title("Frequency of order by hour of day", fontsize=15)
plt.show()


# In[ ]:


We can see that majority of the orders are made during day time. Most of the ordres are placed during time from 9:00 to 18:00 .


# In[ ]:


#### Most Ordered Products


# In[ ]:


We will now look at the products which are in high demand.


# In[89]:


#top25products
top_prod = cart_merged['product_name'].value_counts().reset_index().head(25)
top_prod.columns = ['product_name', 'frequency_count']
top_prod


# Surprising! Most of the top ordered products are organic. Majority of them being fruits.

# Let us try to find out which are the top products ordered in each department.

# In[94]:


#topproductsperdept
top_dept_prod = cart_merged[['order_id', 'product_name', 'department']].drop_duplicates().groupby(by=['department', 'product_name']).agg({'order_id' : ['count']}).reset_index()
top_dept_prod.columns = ['department', 'product_name','orders_count']
top_dept_prod.sort_values(by='orders_count', ascending=False).groupby(by= ['department'], axis=0).head(1)


# Produce tops the list!

# Top products in each aisle would also be a good find.

# In[96]:


#topproductsperaisle
top_aisle_prod = cart_merged[['order_id', 'product_name', 'aisle']].drop_duplicates().groupby(by=['aisle', 'product_name']).agg({'order_id' : ['count']}).reset_index()
top_aisle_prod.columns = ['aisle', 'product_name','orders_count']
top_aisle_prod.sort_values(by='orders_count', ascending=False).groupby(by= ['aisle'], axis=0).head(1).head(10)


# In[ ]:


We can see that fresh fruits are the mostly ordered product.


# In[ ]:


#### Product Reordering


# In[ ]:


We are more concerned about a product being re ordered. We will have a look at how frequently are customers reordering.


# In[87]:


orders.sort_values(by=['days_since_prior_order'])


# In[88]:


#frequencyoforder
plt.figure(figsize=(12,8))
sns.countplot(x="days_since_prior_order", data=orders, color='gray')
plt.ylabel('Count', fontsize=12)
plt.xlabel('Days since prior order', fontsize=12)
plt.xticks(rotation='vertical')
plt.title("Frequency distribution by days since prior order", fontsize=15)
plt.show()


# Looks like customers order once in every week (peak at 7 days) or once in a month (peak at 30 days). We could also see smaller peaks at 14, 21 and 28 days (weekly intervals)

# In[ ]:


Let us calculate the re-order percentage.


# In[97]:


# percentage of re-orders 
cart_merged.reordered.sum() / cart_merged.shape[0]


# On an average, about 59% of the products in an order are re-ordered products

# Now let us check the reordered percentage of each department.

# In[98]:


dep_reorder = cart_merged.groupby(["department"])["reordered"].aggregate("mean").reset_index()

plt.figure(figsize=(12,8))
sns.pointplot(dep_reorder['department'].values, dep_reorder['reordered'].values, alpha=0.8, color='blue')
plt.ylabel('Reorder ratio', fontsize=12)
plt.xlabel('Department', fontsize=12)
plt.title("Department wise reorder ratio", fontsize=15)
plt.xticks(rotation='vertical')
plt.show()


# In[ ]:


We can see that Personal care has lowest reorder ratio and dairy eggs have highest reorder ratio.


# Let us now explore the relationship between how order of adding the product to the cart affects the reorder ratio.

# In[99]:


cart_reorder = cart_merged[(cart_merged.add_to_cart_order <= 70)]
cart_reorder = cart_reorder.groupby(['add_to_cart_order']).agg({'reordered' : ['mean']}).reset_index()
cart_reorder.columns = ['add_to_cart_order','re_order_percent']
plt.figure(figsize=(12,8))
sns.pointplot(cart_reorder['add_to_cart_order'].values, cart_reorder['re_order_percent'].values, alpha=0.8, color='blue')
plt.ylabel('Reorder Percent', fontsize=13)
plt.xlabel('Add to cart order', fontsize=13)
plt.title("Relation between Add to cart order and Reorder ratio")
plt.xticks(rotation='vertical')
plt.show()


# It looks like the products that are added to the cart initially are more likely to be reordered again compared to the ones added later. This makes sense since we tend to first order all the products we used to buy frequently and then look out for the new products available.

# We will now check what day of the week and what time are people reordering products.

# In[100]:


#reorderbydowandhod
dow_hod_reorder = cart_merged.groupby(["order_dow", "order_hour_of_day"])["reordered"].aggregate("mean").reset_index()
dow_hod_reorder = dow_hod_reorder.pivot('order_dow', 'order_hour_of_day', 'reordered')

plt.figure(figsize=(12,6))
sns.heatmap(dow_hod_reorder)
plt.title("Reorder ratio of Day of week Vs Hour of day")
plt.show()


# It looks like reorder ratios are quite high during the early mornings compared to later half of the day.

# ### Association Rules Mining

# We will implement association rules i.e. understand what product combinations occur the most. These insights will feed into the recommenders such as "Frequently bought together", "Customer who purchased item A also bought item B" etc.
# 
# We will use the order_products table for this as we will have to prepare the data in a format such that we just have order information i.e. order id and the corresponding product ids in the order. We will join back the product names after we form the association rules

# In[102]:


#ARM
orders_arm = order_products_train.set_index('order_id')['product_id'].rename('item_id')
display(orders_arm.head(10))
type(orders_arm)


# We will define the help functions for association mining

# In[113]:


# Returns frequency counts for items and item pairs
def freq(iterable):
    if type(iterable) == pd.core.series.Series:
        return iterable.value_counts().rename("freq")
    else: 
        return pd.Series(Counter(iterable)).rename("freq")

    
# Returns number of unique orders
def order_count(order_item):
    return len(set(order_item.index))


# Returns generator that yields item pairs, one at a time
def get_item_pairs(order_item):
    order_item = order_item.reset_index().to_numpy()
    for order_id, order_object in groupby(order_item, lambda x: x[0]):
        item_list = [item[1] for item in order_object]
              
        for item_pair in combinations(item_list, 2):
            yield item_pair
            

# Returns frequency and support associated with item
def merge_item_stats(item_pairs, item_stats):
    return (item_pairs
                .merge(item_stats.rename(columns={'freq': 'freqA', 'support': 'supportA'}), left_on='item_A', right_index=True)
                .merge(item_stats.rename(columns={'freq': 'freqB', 'support': 'supportB'}), left_on='item_B', right_index=True))


# Returns name associated with item
def merge_item_name(rules, item_name):
    columns = ['itemA','itemB','freqAB','supportAB','freqA','supportA','freqB','supportB', 
               'confidenceAtoB','confidenceBtoA','lift']
    rules = (rules
                .merge(item_name.rename(columns={'item_name': 'itemA'}), left_on='item_A', right_on='item_id')
                .merge(item_name.rename(columns={'item_name': 'itemB'}), left_on='item_B', right_on='item_id'))
    return rules[columns]    


# Defining association rules functions:

# In[114]:


#ARM functions
def association_rules(order_item, min_support):

    print("Starting order_item: {:22d}".format(len(order_item)))


    # Calculate item frequency and support
    item_stats             = freq(order_item).to_frame("freq")
    item_stats['support']  = item_stats['freq'] / order_count(order_item) * 100


    # Filter from order_item items below min support 
    qualifying_items       = item_stats[item_stats['support'] >= min_support].index
    order_item             = order_item[order_item.isin(qualifying_items)]

    print("Items with support >= {}: {:15d}".format(min_support, len(qualifying_items)))
    print("Remaining order_item: {:21d}".format(len(order_item)))


    # Filter from order_item orders with less than 2 items
    order_size             = freq(order_item.index)
    qualifying_orders      = order_size[order_size >= 2].index
    order_item             = order_item[order_item.index.isin(qualifying_orders)]

    print("Remaining orders with 2+ items: {:11d}".format(len(qualifying_orders)))
    print("Remaining order_item: {:21d}".format(len(order_item)))

 # Recalculate item frequency and support
    item_stats             = freq(order_item).to_frame("freq")
    item_stats['support']  = item_stats['freq'] / order_count(order_item) * 100


    # Get item pairs generator
    item_pair_gen          = get_item_pairs(order_item)


    # Calculate item pair frequency and support
    item_pairs              = freq(item_pair_gen).to_frame("freqAB")
    item_pairs['supportAB'] = item_pairs['freqAB'] / len(qualifying_orders) * 100

    print("Item pairs: {:31d}".format(len(item_pairs)))


    # Filter from item_pairs those below min support
    item_pairs              = item_pairs[item_pairs['supportAB'] >= min_support]

    print("Item pairs with support >= {}: {:10d}\n".format(min_support, len(item_pairs)))

    # Create table of association rules and compute relevant metrics
    item_pairs = item_pairs.reset_index().rename(columns={'level_0': 'item_A', 'level_1': 'item_B'})
    item_pairs = merge_item_stats(item_pairs, item_stats)
    
    item_pairs['confidenceAtoB'] = item_pairs['supportAB'] / item_pairs['supportA']
    item_pairs['confidenceBtoA'] = item_pairs['supportAB'] / item_pairs['supportB']
    item_pairs['lift']           = item_pairs['supportAB'] / (item_pairs['supportA'] * item_pairs['supportB'])
    
    
    # Return association rules sorted by lift in descending order
    return item_pairs.sort_values('lift', ascending=False)


#  We are forming association rules having support greater than 1%

# In[115]:


get_ipython().run_cell_magic('time', '', 'rules = association_rules(orders_arm, 0.01)  ')


# We will join the item id with its name and then display the association rules formed.

# In[116]:


# Replace item ID with item name and display association rules
products_name = products.rename(columns={'product_id':'item_id', 'product_name':'item_name'})
rules_final = merge_item_name(rules, products_name).sort_values('lift', ascending=False)
rules_final.head(50)


# From the output above, we see that the top associations are not surprising, with one flavor of an item being purchased with another flavor from the same item family (eg: Strawberry Chia Cottage Cheese with Blueberry Acai Cottage Cheese, Chicken Cat Food with Turkey Cat Food, etc).
# We can use this result to recommend products to customers.

# ### Customer Segmentation - PCA and K-Means Clustering Analysis

# Let us try to find possible clusters among the different customers and substitute single user_id with the cluster to which they are assumed to belong. Hope this would eventually increase the next prediction model performance.
# 
# Ths first thing to do is creating a dataframe with all the purchases made by each user.

# In[ ]:


#Customer Segmentation
aisle_cluster = pd.crosstab(cart_merged['user_id'], cart_merged['aisle'])
aisle_cluster.head(10)


# We can then execute a Principal Component Analysis to the obtained dataframe. This will reduce the number of features from the number of aisles to 6, the numbr of principal components we have chosen.

# In[118]:


from sklearn.decomposition import PCA
pca_aisle = PCA(n_components=6)
pca_aisle_trans = pca_aisle.fit_transform(aisle_cluster)
print("Ratio of variance explained by each PC:",pca_aisle.explained_variance_ratio_)


# As we can see from the explained variance ratio, Component 1 is responsible for 48% of the variance and contains the most information. 
# Let us visualize the variation explained by number of components.

# In[125]:


plt.figure(figsize=(10,8))
plt.plot(range(0,6),pca_aisle.explained_variance_ratio_.cumsum(),marker = 'o',linestyle = '--')
plt.title('Explained Variance by Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')


# Let us check how much of variability is explained by number of PCA components = 6

# In[120]:


pca_aisle.explained_variance_ratio_.sum()


# With 6 principal components, we are able to explain about 72% of the variation in the dataset.

# In[127]:


ps = pd.DataFrame(pca_aisle_trans)
ps.head()


# #### K-means Clsutering

# We will now combine PCA and K-means to segment our data, where we use the scores obtained by the PCA for the fit.

# In[128]:


from sklearn.cluster import KMeans
wcss = []
for i in range (1,10):
    kmeans_pca = KMeans(n_clusters = i, init = 'k-means++',random_state = 42)
    kmeans_pca.fit(pca_aisle_trans)
    wcss.append(kmeans_pca.inertia_)


# In[ ]:


We will use the elbow method to choose the number of clusters for K-means Clustering.


# In[129]:


plt.figure(figsize=(10,8))
plt.plot(range(1,10),wcss,marker = 'o',linestyle = '--')
plt.title('Number of clusters')
plt.xlabel('WCSS')
plt.ylabel('K-means with PCA Clustering')


# In this instance, we can see a kink coming at the 3 clusters mark. So, we’ll be keeping a 3-cluster solution.

# Subsequently, we fit the model with the principal component scores.

# In[132]:


kmeans_pca = KMeans(n_clusters = 3, init = 'k-means++',random_state = 42)
array_aisle_cluster = kmeans_pca.fit_predict(pca_aisle_trans)


# In[133]:


#Visualize it.

label_color_mapping = {0:'r', 1: 'g', 2: 'b',3:'c' , 4:'m'}
label_color = [label_color_mapping[l] for l in array_aisle_cluster]

#Scatterplot showing the cluster to which each user_id belongs.
plt.figure(figsize = (15,8))
plt.scatter(pca_aisle_trans[:,0],pca_aisle_trans[:,1], c= label_color, alpha=0.3) 
plt.show()


# Here is how our clusters appear.We have found a possible clustering for our customers. 
# Let's check if we also manage to find some interesting pattern beneath it.Let us check how many users fall into each of the clusters

# In[134]:


aisle_cluster['cluster'] = array_aisle_cluster


# In[135]:


aisle_cluster['cluster'].value_counts().sort_values(ascending = False)


# In[136]:


aisle_cluster


# Let's check out what are the top 10 goods bought by people of each cluster. 

# In[138]:


cluster0 = aisle_cluster[aisle_cluster['cluster']==0].drop('cluster',axis=1).mean()
cluster1 = aisle_cluster[aisle_cluster['cluster']==1].drop('cluster',axis=1).mean()
cluster2 = aisle_cluster[aisle_cluster['cluster']==2].drop('cluster',axis=1).mean()


# In[139]:


cluster0.sort_values(ascending=False)[0:10]


# In[140]:


cluster1.sort_values(ascending=False)[0:10]


# In[141]:


cluster2.sort_values(ascending=False)[0:10]


# A first analysis of the clusters confirm the initial hypothesis that:
# 
# fresh fruits
# fresh vegetables
# packaged vegetables fruits
# yogurt
# packaged cheese
# milk
# water seltzer sparkling water
# chips pretzels
# are products which are genereically bought by the majority of the customers.
# 
# What we can inspect here is if clusters differ in quantities and proportions, with respect of these goods, or if a cluster is characterized by some goods not included in this list.

# It seems people of cluster 1 buy more fresh vegetables and fresh fruits than the other clusters. As shown by absolute data, Cluster 1 is also the cluster including those customers buying far more goods than any others.
# 
# People of cluster 0 buy more icecream which is not a characteristic of other clusters.
# 
# Absolute Data shows us People of cluster 1 buy a lot of 'Baby Food Formula' which is not even listed in the top 10 products but mainly characterize this cluster. Coherently (I think) with this observation they buy more milk than the others.

#  think another interesting information my come by lookig at the 10th to 15th most bought products for each cluster which will not include the generic products (i.e. vegetables, fruits, water, etc.) bought by anyone.

# In[142]:


cluster0.sort_values(ascending=False)[10:15]


# In[143]:


cluster1.sort_values(ascending=False)[10:15]


# In[144]:


cluster2.sort_values(ascending=False)[10:15]


# In[ ]:


As you can note by taking into account more products clusters start to differ significantly.


# ### Conclusion
# 
# After we have done a detailed exploration on the customer purchase patterns. We built 5 models to primarily summarize and predict future customer orders.
# 
# + Association Rules - This will help in product recommendations i.e. based on the historical trasactions, this will tell us what product could be suggested if the customer adds product A to his cart
# + Principal Component Analysis - This will tell us how can we reduce the dimensionality of the dataset while retaining the percentage of variability being explained. This will feed in as an input into the clustering analysis where we decide on the number of clusters
# + Customer Segmentation - We use k-means clustering to group customers into 3 clusters based on their ordering pattern from aisles

# In[ ]:




