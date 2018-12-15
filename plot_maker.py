from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

def make_plot(fname,name_x,name_y,lines=None,constants=None):
	plt.clf()
	df = pd.read_csv(fname)
	if constants != None:
		df=df.loc[(df[list(constants)] == pd.Series(constants)).all(axis=1)]
	class_x=sorted(df[name_x].unique())
	class_y=sorted(df[name_y].unique())
	arr=np.zeros((len(class_y),len(class_x)))
	for i,x in enumerate(class_x):
		for j,y in enumerate(class_y):
			arr[j,i]=df.where(df[name_x]==x).where(df[name_y]==y)['mean_test_score'].dropna()
	f, ax = plt.subplots(1, 1, figsize=(8, 5))
	cax =ax.matshow(arr, interpolation='nearest')
	f.colorbar(cax)
	ax.set_xlabel(name_x[len('param_clf__'):])
	ax.set_ylabel(name_y[len('param_clf__'):])
	ax.set_yticklabels(['']+['{:.2e}'.format(x)  for x in class_y])
	ax.set_xticklabels(['{:.2e}'.format(x) for x in np.array(class_x).astype(float)])
	plt.show()

def make_lines(fname,name_x,name_y,constants=None):
	plt.clf()
	df = pd.read_csv(fname)
	if constants != None:
		df=df.loc[(df[list(constants)] == pd.Series(constants)).all(axis=1)]
	df = pd.read_csv(fname).sort_values(by=name_x)
	Xs=df.groupby(name_y)[name_x].apply(list)
	Ys=df.groupby(name_y)['mean_test_score'].apply(list)
	for y in df[name_y].unique():
		plt.plot(Xs[y],Ys[y],label=y)
	plt.legend()
	plt.xlabel(name_x[len('param_clf__'):])
	plt.ylabel('Mean Testing Score')
	plt.show()
def make_hist(fname,name_x):
	df=pd.read_csv(fname)[['Value','Frequency','Probability']]
	df=df.set_index('Value')
	f, ax = plt.subplots(2, 1, figsize=(8, 5))
	ax[0].hist(df['Frequency'].tolist(),bins=df.index.tolist())
	ax[1].hist(df['Probability'].tolist(),bins=df.index.tolist())
	f.show()