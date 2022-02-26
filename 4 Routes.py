#!/usr/bin/env python
# coding: utf-8

# # read data

# In[1]:


# load libraries
import glob
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#import statsmodels.api as sm


# In[2]:


# display all columns
pd.options.display.max_columns = 50
#pd.options.display.max_rows = None


# In[3]:


# get lists of multiple json files
prep_data = pd.read_json("data/A20runs.json")
prep_data.head()


# In[4]:


print(prep_data['path_taken'][0])


# In[5]:


prep_data.shape


# # Get path taken
# - elite
# - store
# - monster
# - rest
# - random

# In[6]:


# count campfire lift
def path_e(row):
    cnt = 0
    for element in row:
        if element == 'E':
            cnt = cnt + 1    
    return cnt
prep_data['path_taken_elite'] = prep_data['path_taken'].apply(path_e)


# In[7]:


# count campfire lift
def path_s(row):
    cnt = 0
    for element in row:
        if element == '$':
            cnt = cnt + 1    
    return cnt
prep_data['path_taken_store'] = prep_data['path_taken'].apply(path_s)


# In[8]:


# count campfire lift
def path_r(row):
    cnt = 0
    for element in row:
        if element == 'R':
            cnt = cnt + 1    
    return cnt
prep_data['path_taken_rest'] = prep_data['path_taken'].apply(path_r)


# In[9]:


# count campfire lift
def path_rand(row):
    cnt = 0
    for element in row:
        if element == '?':
            cnt = cnt + 1    
    return cnt
prep_data['path_taken_random'] = prep_data['path_taken'].apply(path_rand)


# In[10]:


# count campfire lift
def path_m(row):
    cnt = 0
    for element in row:
        if element == 'M':
            cnt = cnt + 1    
    return cnt
prep_data['path_taken_monster'] = prep_data['path_taken'].apply(path_m)


# In[11]:


# count campfire lift
def path_t(row):
    cnt = 0
    for element in row:
        if element == 'T':
            cnt = cnt + 1    
    return cnt
prep_data['path_taken_treasure'] = prep_data['path_taken'].apply(path_t)


# # Analytical Modelling
# - to visualize path taken dataframe

# In[12]:


temp = prep_data.copy()
temp['unknowns'] = temp['path_taken'].apply(path_rand)
temp['merchants'] = temp['path_taken'].apply(path_s)
temp['treasures'] = temp['path_taken'].apply(path_t)
temp['rests'] = temp['path_taken'].apply(path_r)
temp['enemies'] = temp['path_taken'].apply(path_m)
temp['elites'] = temp['path_taken'].apply(path_e)
temp[['unknowns','merchants','treasures','rests','enemies','elites']].head()


# # Cluster Path Taken

# In[13]:


paths = prep_data[['path_taken_elite', 'path_taken_store', 'path_taken_rest', 'path_taken_random', 'path_taken_monster', 'path_taken_treasure', 'character_chosen']] 
#paths.index = prep_data['character_chosen']
paths.head()


# In[14]:


from sklearn.cluster import KMeans
from sklearn import metrics

kmeans_kwargs = {
     "init": "random",
     "n_init": 10,
     "max_iter": 300,
     "random_state": 42,
 }

# A list holds the SSE values for each k
sse = []
sh = []
ch = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    # wsse
    kmeans.fit(paths.iloc[:,0:-1])
    sse.append(kmeans.inertia_)
    # ch score
    if k == 1: # skips k = 1 to prevent error
        continue
    labels = kmeans.labels_
    sh.append(metrics.silhouette_score(paths.iloc[:,0:-1], labels, metric = 'euclidean'))
    ch.append(metrics.calinski_harabasz_score(paths.iloc[:,0:-1], labels))


# In[15]:


plt.style.use('default')

lvl = 20

fig, ax = plt.subplots()
ax.plot(range(1, 11), sse)

ax.set_xticks(range(1, 11))
ax.set_xlabel("Number of Clusters")
ax.set_ylabel("SSE")
ax.set_title('Cluster Paths Taken beyond Level {}'.format(lvl))
plt.show()


# In[16]:


plt.style.use('default')

lvl = 20

fig, ax = plt.subplots()
ax.plot(range(2, 11), ch)

ax.set_xticks(range(1, 11))
ax.set_xlabel("Number of Clusters")
ax.set_ylabel("CH Score")
ax.set_title('Cluster Paths Taken beyond Level {}'.format(lvl))
plt.show()


# In[17]:


plt.style.use('default')

lvl = 20

fig, ax = plt.subplots()
ax.plot(range(2, 11), sh)

ax.set_xticks(range(1, 11))
ax.set_xlabel("Number of Clusters")
ax.set_ylabel("Silhouette Score")
ax.set_title('Cluster Paths Taken beyond Level {}'.format(lvl))
plt.show()


# In[18]:


paths.head()


# **2 is the optimal number of clusters.** Thus, there is 2 strategy to win STS Ascension Runs

# In[19]:


kmeans = KMeans(n_clusters = 2, random_state= 123)
paths['cluster'] = kmeans.fit_predict(paths.iloc[:,0:-1])


# In[20]:


paths.head()


# In[21]:


paths.groupby('cluster').size()


# In[22]:


# We can just extend this dictionary to aggregate by multiple functions or multiple columns.
paths.groupby("cluster").agg({"path_taken_elite":['mean', 'std'],
                              "path_taken_store":['mean', 'std'],
                              "path_taken_rest":['mean', 'std'],
                              "path_taken_random":['mean', 'std'],
                              "path_taken_monster":['mean', 'std'],
                              "path_taken_treasure":['mean', 'std'],
                              "cluster": ['count']})


# In[23]:


paths.groupby(["character_chosen","cluster"]).agg({"cluster": ['count']})


# In[24]:


paths.groupby("cluster").quantile([0.1, 0.5, 0.9])


# In[25]:


paths.quantile([0.1, 0.5, 0.9])


# # Plot Functions

# In[26]:


def plot_hist(df, fig_w = 5, fig_h = 3.5, hist_col = 'grey', edge_col = 'white',
              xlab = "Number of Elite Fights", ylab = "Run Count",
              title = "Elites | Path Taken", title_y = 1.15, title_x = -0.16, title_col = 'black', title_size = 18,
              subtit = "Ideal No. of Elites: 7-10", subtit_y = 1650, subtit_x = -2.59, subtit_size = 12, subtit_col = 'black',
              ax_col = 'silver', savfig = "img/path_elites.png"):
    #df 

    plt.style.use('default')

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.hist(df, bins=np.arange(df.min(), df.max()+1)-0.5, color = hist_col, edgecolor = edge_col)
    ax.set_xticks(np.arange(0, df.max()+1, 1))

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab, loc = 'top')

    # modify title fonts and location
    ax.set_title(title, loc = 'left', y = title_y, x = title_x, 
                         fontdict = {'family': 'sans-serif',
                                    'color':  title_col,
                                    'weight': 'bold',
                                    'size': title_size
                                    })

    # add subtitle text (very manual on x y coordinate)
    ax.text(s = subtit, y = subtit_y, x = subtit_x, 
                         fontdict = {'family': 'sans-serif',
                                    'color':  subtit_col,
                                    'size': subtit_size
                                    }) 


    # remove spines top and right
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # remove xticks
    ax.xaxis.set_ticks_position('none') 

    # change color of axis, ticks
    ax_tick_col = ax_col
    ax.spines['bottom'].set_color(ax_tick_col)
    ax.spines['left'].set_color(ax_tick_col)

    ax.xaxis.label.set_color(ax_tick_col)
    ax.yaxis.label.set_color(ax_tick_col)
    ax.tick_params(axis='y', colors= ax_tick_col)
    ax.tick_params(axis='x', colors= ax_tick_col)


    plt.tight_layout()

    plt.show()
    
    fig.savefig(savfig)
    


# In[27]:


def plot_hist_2clus(df_1st, df_2nd, df,
                    df_1st_col = 'black', df_2nd_col = 'blue',
                  fig_w = 5, fig_h = 3.5, hist_col = 'grey', edge_col = 'white',
                  xlab = "Number of Unknowns Taken", ylab = "Run Count",
                  title = "Unknowns | Path Taken", title_y = 1.15, title_x = -0.16, title_col = 'black', title_size = 18,
                  df_1st_s = "Ideal No. of Unknowns: 7-10", df_1st_y = 1010, df_1st_x = -2.59, df_1st_size = 12,
                  df_2nd_s = "Ideal No. of Unknowns: 7-10", df_2nd_y = 1010, df_2nd_x = 12.59, df_2nd_size = 12,
                  ax_col = 'silver', savfig = "img/path_random_cluster.png"):
        #df 

    plt.style.use('default')

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.hist([df_1st, df_2nd], bins=np.arange(df.min(), df.max()+1)-0.5, edgecolor = edge_col, color = [df_1st_col, df_2nd_col])
    ax.set_xticks(np.arange(0, df.max()+1, 1))

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab, loc = 'top')

    # modify title fonts and location
    ax.set_title(title, loc = 'left', y = title_y, x = title_x, 
                         fontdict = {'family': 'sans-serif',
                                    'color':  title_col,
                                    'weight': 'bold',
                                    'size': title_size
                                    })

    # add subtitle text (very manual on x y coordinate)
    ax.text(s = df_1st_s, y = df_1st_y, x = df_1st_x, 
                         fontdict = {'family': 'sans-serif',
                                    'color':  df_1st_col,
                                    'size': df_1st_size
                                    }) 
    
    # add subtitle text (very manual on x y coordinate)
    ax.text(s = df_2nd_s, y = df_2nd_y, x = df_2nd_x, 
                         fontdict = {'family': 'sans-serif',
                                    'color':  df_2nd_col,
                                    'size': df_2nd_size
                                    }) 


    # remove spines top and right
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # remove xticks
    ax.xaxis.set_ticks_position('none') 

    # change color of axis, ticks
    ax_tick_col = ax_col
    ax.spines['bottom'].set_color(ax_tick_col)
    ax.spines['left'].set_color(ax_tick_col)

    ax.xaxis.label.set_color(ax_tick_col)
    ax.yaxis.label.set_color(ax_tick_col)
    ax.tick_params(axis='y', colors= ax_tick_col)
    ax.tick_params(axis='x', colors= ax_tick_col)


    plt.tight_layout()

    plt.show()
    
    fig.savefig(savfig)
    


# # Plot Output

# ### Elite Paths Taken

# In[28]:


plot_hist(prep_data["path_taken_elite"], fig_w = 5, fig_h = 3.5, hist_col = 'grey', edge_col = 'white',
          xlab = "Number of Elite Taken", ylab = "Run Count",
          title = "Elites | Path Taken", title_y = 1.15, title_x = -0.16, title_col = 'black', title_size = 18,
          subtit = "Ideal No. of Elites: 6-11", subtit_y = 1650, subtit_x = -2.59, subtit_size = 12, subtit_col = 'black',
          ax_col = 'silver', savfig = "img/path_elites.png")


# In[29]:


plot_hist_2clus(df_1st = paths["path_taken_elite"].loc[paths["cluster"] == 0], 
                df_2nd = paths["path_taken_elite"].loc[paths["cluster"] == 1],
                df = prep_data["path_taken_elite"],
                df_1st_col = sns.color_palette("Dark2", 8)[0], df_2nd_col = sns.color_palette("Dark2", 8)[1],
                fig_w = 5, fig_h = 3.5, hist_col = 'grey', edge_col = 'white',
                xlab = "Number of Elites Taken", ylab = "Run Count",
                title = "Elite | Path Taken", title_y = 1.15, title_x = -0.14, title_col = 'black', title_size = 18,
                df_1st_s = "More Enemies: 6-11", df_1st_y = 860, df_1st_x = 0, df_1st_size = 12,
                df_2nd_s = "More Unknowns: 6-10", df_2nd_y = 860, df_2nd_x = 8, df_2nd_size = 12,
                ax_col = 'silver', savfig = "img/path_elites_cluster.png")


# ### Store Paths Taken

# In[30]:


plot_hist(prep_data["path_taken_store"], fig_w = 5, fig_h = 3.5, hist_col = 'grey', edge_col = 'white',
          xlab = "Number of Merchants Taken", ylab = "Run Count",
          title = "Merchants | Path Taken", title_y = 1.15, title_x = -0.16, title_col = 'black', title_size = 18,
          subtit = "Ideal No. of Merchants: 3-6", subtit_y = 2650, subtit_x = -1.59, subtit_size = 12, subtit_col = 'black',
          ax_col = 'silver', savfig = "img/path_stores.png")


# In[31]:


plot_hist_2clus(df_1st = paths["path_taken_store"].loc[paths["cluster"] == 0], 
                df_2nd = paths["path_taken_store"].loc[paths["cluster"] == 1],
                df = prep_data["path_taken_store"],
                df_1st_col = sns.color_palette("Dark2", 8)[0], df_2nd_col = sns.color_palette("Dark2", 8)[1],
                fig_w = 5, fig_h = 3.5, hist_col = 'grey', edge_col = 'white',
                xlab = "Number of Merchants Taken", ylab = "Run Count",
                title = "Merchants | Path Taken", title_y = 1.15, title_x = -0.14, title_col = 'black', title_size = 18,
                df_1st_s = "More Enemies: 3-6", df_1st_y = 1350, df_1st_x = 0, df_1st_size = 12,
                df_2nd_s = "More Unknowns: 3-6", df_2nd_y = 1350, df_2nd_x = 5, df_2nd_size = 12,
                ax_col = 'silver', savfig = "img/path_stores_cluster.png")


# ### Campfire Paths Taken (Rest)

# In[32]:


plot_hist(df = prep_data["path_taken_rest"], fig_w = 5, fig_h = 3.5, hist_col = 'grey', edge_col = 'white',
          xlab = "Number of Rests Taken", ylab = "Run Count",
          title = "Rests | Path Taken", title_y = 1.15, title_x = -0.16, title_col = 'black', title_size = 18,
          subtit = "Ideal No. of Rests: 7-11", subtit_y = 1900, subtit_x = -2.59, subtit_size = 12, subtit_col = 'black',
          ax_col = 'silver', savfig = "img/path_campfires.png")


# In[33]:


plot_hist_2clus(df_1st = paths["path_taken_rest"].loc[paths["cluster"] == 0], 
                df_2nd = paths["path_taken_rest"].loc[paths["cluster"] == 1],
                df = prep_data["path_taken_rest"],
                df_1st_col = sns.color_palette("Dark2", 8)[0], df_2nd_col = sns.color_palette("Dark2", 8)[1],
                fig_w = 5, fig_h = 3.5, hist_col = 'grey', edge_col = 'white',
                xlab = "Number of Rests Taken", ylab = "Run Count",
                title = "Rests | Path Taken", title_y = 1.15, title_x = -0.14, title_col = 'black', title_size = 18,
                df_1st_s = "More Enemies: 7-11", df_1st_y = 970, df_1st_x = 0, df_1st_size = 12,
                df_2nd_s = "More Unknowns: 7-11", df_2nd_y = 970, df_2nd_x = 8, df_2nd_size = 12,
                ax_col = 'silver', savfig = "img/path_campfires_cluster.png")


# ### Random Paths Taken

# In[34]:


plot_hist(df = prep_data["path_taken_random"], fig_w = 5, fig_h = 3.5, hist_col = 'grey', edge_col = 'white',
          xlab = "Number of Unknowns Taken", ylab = "Run Count",
          title = "Unknowns | Path Taken", title_y = 1.15, title_x = -0.16, title_col = 'black', title_size = 18,
          subtit = "Ideal No. of Unknowns: 8-14", subtit_y = 1240, subtit_x = -3.2, subtit_size = 12, subtit_col = 'black',
          ax_col = 'silver', savfig = "img/path_random.png")


# In[35]:


plot_hist_2clus(df_1st = paths["path_taken_random"].loc[paths["cluster"] == 0], 
                df_2nd = paths["path_taken_random"].loc[paths["cluster"] == 1],
                df = prep_data["path_taken_random"],
                df_1st_col = sns.color_palette("Dark2", 8)[0], df_2nd_col = sns.color_palette("Dark2", 8)[1],
                fig_w = 5, fig_h = 3.5, hist_col = 'grey', edge_col = 'white',
                xlab = "Number of Unknowns Taken", ylab = "Run Count",
                title = "Unknowns | Path Taken", title_y = 1.15, title_x = -0.14, title_col = 'black', title_size = 18,
                df_1st_s = "More Enemies: 7-11", df_1st_y = 1010, df_1st_x = 0, df_1st_size = 12,
                df_2nd_s = "More Unknowns: 11-15", df_2nd_y = 1010, df_2nd_x = 10, df_2nd_size = 12,
                ax_col = 'silver', savfig = "img/path_random_cluster.png")


# ### Monster Paths Taken

# In[36]:


plot_hist(df = prep_data["path_taken_monster"], fig_w = 5, fig_h = 3.5, hist_col = 'grey', edge_col = 'white',
          xlab = "Number of Enemies Taken", ylab = "Run Count",
          title = "Enemies | Path Taken", title_y = 1.15, title_x = -0.16, title_col = 'black', title_size = 18,
          subtit = "Ideal No. of Enemies: 9-15", subtit_y = 1310, subtit_x = -3.7, subtit_size = 12, subtit_col = 'black',
          ax_col = 'silver', savfig = "img/path_monsters.png")


# In[37]:


plot_hist_2clus(df_1st = paths["path_taken_monster"].loc[paths["cluster"] == 0], 
                df_2nd = paths["path_taken_monster"].loc[paths["cluster"] == 1],
                df = prep_data["path_taken_monster"],
                df_1st_col = sns.color_palette("Dark2", 8)[0], df_2nd_col = sns.color_palette("Dark2", 8)[1],
                fig_w = 5, fig_h = 3.5, hist_col = 'grey', edge_col = 'white',
                xlab = "Number of Enemies Taken", ylab = "Run Count",
                title = "Enemies | Path Taken", title_y = 1.15, title_x = -0.14, title_col = 'black', title_size = 18,
                df_1st_s = "More Enemies: 12-16", df_1st_y = 1100, df_1st_x = 0, df_1st_size = 12,
                df_2nd_s = "More Unknowns: 9-13", df_2nd_y = 1100, df_2nd_x = 12, df_2nd_size = 12,
                ax_col = 'silver', savfig = "img/path_monsters_cluster.png")


# ### Treasure Paths Taken

# In[38]:


plot_hist(df = prep_data["path_taken_treasure"], fig_w = 5, fig_h = 3.5, hist_col = 'grey', edge_col = 'white',
          xlab = "Number of Treasures Taken", ylab = "Run Count",
          title = "Treasures | Path Taken", title_y = 1.15, title_x = -0.16, title_col = 'black', title_size = 18,
          subtit = "Ideal No. of Treasures: 3", subtit_y = 7900, subtit_x = -0.95, subtit_size = 12, subtit_col = 'black',
          ax_col = 'silver', savfig = "img/path_treasures.png")


# In[39]:


plot_hist_2clus(df_1st = paths["path_taken_treasure"].loc[paths["cluster"] == 0], 
                df_2nd = paths["path_taken_treasure"].loc[paths["cluster"] == 1],
                df = prep_data["path_taken_treasure"],
                df_1st_col = sns.color_palette("Dark2", 8)[0], df_2nd_col = sns.color_palette("Dark2", 8)[1],
                fig_w = 5, fig_h = 3.5, hist_col = 'grey', edge_col = 'white',
                xlab = "Number of Treasures Taken", ylab = "Run Count",
                title = "Treasures | Path Taken", title_y = 1.15, title_x = -0.14, title_col = 'black', title_size = 18,
                df_1st_s = "More Enemies: 3", df_1st_y = 4100, df_1st_x = 0, df_1st_size = 12,
                df_2nd_s = "More Unknowns: 3", df_2nd_y = 4100, df_2nd_x = 3, df_2nd_size = 12,
                ax_col = 'silver', savfig = "img/path_treasures_cluster.png")


# # Basic Cluster Count Overall

# In[40]:


paths_cluster_all = paths.groupby('cluster').size()
paths_cluster_all.index = ['More\nEnemies\nPath', 'More\nUnknowns\nPath']


# In[41]:


fig, ax = plt.subplots(figsize=(5, 3))
paths
ax.barh(paths_cluster_all.index, paths_cluster_all, color = [sns.color_palette("Dark2", 8)[0], sns.color_palette("Dark2", 8)[1]])
ax.set_title("2 Ideal Paths | Cluster Count", y = 1.15, x = -0.21,
                                                loc = 'left', fontdict = {'family': 'sans-serif',
                                                'color':  'black',
                                                'weight': 'bold',
                                                'size': 18
                                                })
ax.set_xlabel('Membership Counts')


ax.text(s = "Balanced Cluster Size", y = 1.58, x = -800, 
                         fontdict = {'family': 'sans-serif',
                                    'color':  'grey',
                                    'size': 12
                                    }) 

# remove spines top and right
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

# remove xticks
ax.xaxis.set_ticks_position('none') 

# change color of axis, ticks
ax_tick_col = 'silver'
ax.spines['bottom'].set_color(ax_tick_col)
ax.spines['left'].set_color(ax_tick_col)

ax.xaxis.label.set_color(ax_tick_col)
ax.yaxis.label.set_color(ax_tick_col)
#ax.tick_params(axis='y', colors= ax_tick_col)
ax.tick_params(axis='x', colors= ax_tick_col)



plt.tight_layout()

plt.show()


# In[42]:


fig.savefig('img/path_cluster_size.png')


# # Character Cluster Count

# In[43]:


paths.head()


# In[44]:


paths_cluster_char = paths.groupby(['character_chosen','cluster']).size()
paths_cluster_char =paths_cluster_char.reindex(["IRONCLAD", "THE_SILENT", "DEFECT", "WATCHER"], level = 0)
#paths_cluster_all.index = ['More\nEnemies\nPath', 'More\nUnknowns\nPath']


# In[45]:


paths_cluster_char


# In[46]:


paths_cluster_char = paths_cluster_char.reset_index().rename(columns={0: 'count'})
paths_cluster_char


# In[47]:


df_1st = paths_cluster_char.loc[paths_cluster_char["cluster"] == 0] 
df_1st.index = df_1st['character_chosen']
df_1st = df_1st['count']
df_1st


# In[48]:


df_2nd = paths_cluster_char.loc[paths_cluster_char["cluster"] == 1]
df_2nd.index = df_2nd['character_chosen']
df_2nd = df_2nd['count']
df_2nd


# In[49]:


def subcategorybar(X,  # categories names
                   vals,  # subcategories values
                   base_colors,  # base colors for each subcategory
                   width=0.9):  # total subcategories bars width
    n = len(vals)
    _X = np.arange(len(X))
    for i in range(n):
        # draw i-th subcategory bars
        bars = plt.bar(_X - width / 2. + i / float(n) * width,
                       vals[i],
                       width=width / float(n),
                       align="edge",  # aligns bars by their left edges
                       color=base_colors[i])  # use base color for subcategory
    # set category tick locations and labels of the x-axis
    plt.xticks(_X, X)


# In[50]:


df_1st.values


# In[51]:


fig, ax = plt.subplots(figsize=(5, 3.5))
subcategorybar(['IRONCLAD', 'SILENT', 'DEFECT', 'WATCHER'],  # responses
               [df_1st.values, df_2nd.values],  # pre- and post- responses distribution
               [sns.color_palette("Dark2", 8)[0], sns.color_palette("Dark2", 8)[1]])


ax.set_ylabel('Membership Counts', loc = 'top')

ax.set_title("2 Ideal Paths | Characters", y = 1.15, x = -0.16,
                                                loc = 'left', fontdict = {'family': 'sans-serif',
                                                'color':  'black',
                                                'weight': 'bold',
                                                'size': 18
                                                })

ax.text(s = "Balanced Cluster Size", y = 1540, x = -1.3, 
                         fontdict = {'family': 'sans-serif',
                                    'color':  'black',
                                    'size': 12
                                    }) 


# remove spines top and right
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

# remove xticks
ax.xaxis.set_ticks_position('none') 

# change color of axis, ticks
ax_tick_col = 'silver'
ax.spines['bottom'].set_color(ax_tick_col)
ax.spines['left'].set_color(ax_tick_col)

ax.xaxis.label.set_color(ax_tick_col)
ax.yaxis.label.set_color(ax_tick_col)
ax.tick_params(axis='y', colors= ax_tick_col)
#ax.tick_params(axis='x', colors= ax_tick_col)



plt.tight_layout()

plt.show()


# In[52]:


fig.savefig('img/path_cluster_char.png')


# # Cluster Fig Random

# In[53]:


plot_hist(df = prep_data["path_taken_random"].loc[paths["cluster"] == 0], 
          fig_w = 5, fig_h = 3.5, hist_col = sns.color_palette("Dark2", 8)[0], edge_col = 'white',
          xlab = "Number of Unknowns Taken", ylab = "Run Count",
          title = "More Enemies Path | Unknowns", title_y = 1.15, title_x = -0.13, title_col = sns.color_palette("Dark2", 8)[0], title_size = 18,
          subtit = "Ideal No. of Unknowns: 7-11", subtit_y = 930, subtit_x = -2.3, subtit_size = 12, subtit_col = 'black',
          ax_col = 'silver', savfig = "img/path_random_0.png")


# In[54]:


plot_hist(df = prep_data["path_taken_random"].loc[paths["cluster"] == 1], 
          fig_w = 5, fig_h = 3.5, hist_col = sns.color_palette("Dark2", 8)[1], edge_col = 'white',
          xlab = "Number of Unknowns Taken", ylab = "Run Count",
          title = "More Unknowns Path | Unknowns", title_y = 1.15, title_x = -0.13, title_col = sns.color_palette("Dark2", 8)[1], title_size = 18,
          subtit = "Ideal No. of Unknowns: 11-15", subtit_y = 1020, subtit_x = -2.5, subtit_size = 12, subtit_col = 'black',
          ax_col = 'silver', savfig = "img/path_random_1.png")


# # Cluster Fig Enemies

# In[55]:


plot_hist(df = prep_data["path_taken_monster"].loc[paths["cluster"] == 0], 
          fig_w = 5, fig_h = 3.5, hist_col = sns.color_palette("Dark2", 8)[0], edge_col = 'white',
          xlab = "Number of Enemies Taken", ylab = "Run Count",
          title = "More Enemies Path | Enemies", title_y = 1.15, title_x = -0.13, title_col = sns.color_palette("Dark2", 8)[0], title_size = 18,
          subtit = "Ideal No. of Enemies: 12-16", subtit_y = 900, subtit_x = -2.9, subtit_size = 12, subtit_col = 'black',
          ax_col = 'silver', savfig = "img/path_monster_0.png")


# In[56]:


plot_hist(df = prep_data["path_taken_monster"].loc[paths["cluster"] == 1], 
          fig_w = 5, fig_h = 3.5, hist_col = sns.color_palette("Dark2", 8)[1], edge_col = 'white',
          xlab = "Number of Enemies Taken", ylab = "Run Count",
          title = "More Unknowns Path | Enemies", title_y = 1.15, title_x = -0.16, title_col = sns.color_palette("Dark2", 8)[1], title_size = 18,
          subtit = "Ideal No. of Enemies: 9-13", subtit_y = 1100, subtit_x = -2.4, subtit_size = 12, subtit_col = 'black',
          ax_col = 'silver', savfig = "img/path_monster_1.png")


# In[ ]:





# # View Sample Results per Cluster
# 
# 1 ~ Monster
# 
# 0 ~ Random

# In[57]:


paths.iloc[:, -1].head(6)


# ## Monster Group (1) 

# In[58]:


idx_1 = paths.index[paths['cluster'] == 1].tolist()[10:14]
idx_1


# In[59]:


for i in idx_1:
    print(i)
    print(prep_data['path_taken'][i])


# ## Random Group (0) 

# In[60]:


idx_0 = paths.index[paths['cluster'] == 0].tolist()[10:14]
idx_0


# In[61]:


for i in idx_0:
    print(i)
    print(prep_data['path_taken'][i])

