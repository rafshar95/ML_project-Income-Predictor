from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

def make_plot(fname,name_x,name_y,lines=None):
	plt.clf()
	df = pd.read_csv(fname)
	class_x=df[name_x].unique()
	class_y=df[name_y].unique()
	arr=np.zeros((len(class_x),len(class_y)))
	for i,x in enumerate(class_x):
		for j,y in enumerate(class_y):
			arr[i,j]=df.where(df[name_x]==x).where(df[name_y]==y)['mean_test_score'].dropna()
	f, ax = plt.subplots(1, 1, figsize=(8, 5))
	cax =ax.matshow(arr, interpolation='nearest')
	f.colorbar(cax)
	ax.set_ylabel(name_y[len('param_clf__'):])
	ax.set_xlabel(name_x[len('param_clf__'):])
	ax.set_yticklabels(['{:.2e}'.format(x) for x in class_y.astype(float)])
	ax.set_xticklabels(['{:.2e}'.format(x) for x in class_x.astype(float)])
	plt.savefig(fname+'.graph.pdf')

def make_lines(fname,name_x,name_y):
	plt.clf()
	df = pd.read_csv(fname).sort_values(by=name_x)
	Xs=df.groupby(name_y)[name_x].apply(list)
	Ys=df.groupby(name_y)['mean_test_score'].apply(list)
	for y in df[name_y].unique():
		plt.plot(Xs[y],Ys[y],label=y)
	plt.legend()
	plt.savefig(fname+'.graph.pdf')