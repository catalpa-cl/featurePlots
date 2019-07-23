import seaborn as sns
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import nltk
import re


import matplotlib.pyplot as plt
sns.set(style="ticks")
sns.set_style("whitegrid")

# data = pd.read_csv('data/featureFile.txt', sep='\t');
# g = sns.jointplot(df.outcome,df.NrofTokens,
#                    height=5, ratio=3, color="b",kind="reg")
# g.annotate(stats.pearsonr)
dkprodata = pd.read_csv('data/featureFile2.txt', sep='\t')
dkprodata.head(n=3)
dkprodata.dropna
# Plot formatting
plt.legend(prop={'size': 12})
plt.title('NrofTokens vs Scores')
plt.xlabel('NrofTokens')
plt.ylabel('Density')
df = dkprodata[dkprodata.outcome == 0]
sns.distplot(df['NrofTokens'], hist = False, kde = True, label='Score 0')
df = dkprodata[dkprodata.outcome == 1]
sns.distplot(df['NrofTokens'], hist = False, kde = True, label='Score 1')
df = dkprodata[dkprodata.outcome == 2]
sns.distplot(df['NrofTokens'], hist = False, kde = True, label='Score 2')
df = dkprodata[dkprodata.outcome == 3]
sns.distplot(df['NrofTokens'], hist = False, kde = True, label='Score 3')
df = dkprodata[dkprodata.outcome == 4]
sns.distplot(df['NrofTokens'], hist = False, kde = True, label='Score 4')
df = dkprodata[dkprodata.outcome == 5]
sns.distplot(df['NrofTokens'], hist = False, kde = True, label='Score 5')
plt.show()