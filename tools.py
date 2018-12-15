import numpy as np
import pandas as pd

def get_frequencies(data):
	for col in data.columns:
		counts = data[col].value_counts(dropna=False).sort_index()
		counts.plot(kind='bar',axes=plt)
		labels = counts.keys()
		prob = data.groupby(col).apply(lambda x: x[col].where(x['income']==' >50K').count()/x['income'].count()).sort_index()
		ig = data.groupby(col).apply(lambda x: x[col].where(x['income']==' >50K').count()/x['income'].count()).sort_index()	
		test=[a==b for a,b in zip(prob.keys(),counts.keys())]
		np.savetxt('feature_'+col+'.cts',np.swapaxes([labels,counts,prob],0,1),"%s,%.3f,%.3f",header='Value,Frequency,Probability', comments='', delimiter=',')

##work in progress:
def calculate_IG(data,field,to_merge):
	## to_merge is a dictionary with old feature as key and new feature as value
	new_groups = np.unique([v for v in to_merge.values()])
	Hold = np.zeros(len(new_groups))
	IGold = np.zeros(len(new_groups))
	merged = data.copy()
	for k,v in to_merge.items():
		i=np.where(new_groups==v)
		A,B=calc_entropy(data,field,k)
		print(A,B)
		Hold[i]=Hold[i]+A
		IGold[i]=IGold[i]+B
	Hnew = np.zeros(len(new_groups))
	IGnew = np.zeros(len(new_groups))
	
	merged = merged.replace(to_merge)
	for i,g in enumerate(new_groups):
		Hnew[i],IGnew[i]=calc_entropy(merged,field,g)
		print(g,Hnew[i],IGnew[i])
	old=-Hold+IGold
	new=-Hnew+IGnew
	return old,new

def calc_entropy(data,field,value):
	p50=(data.where(data['income']==' >50K').count()/data.count()).iloc[0]
	print(p50)
	p50k=(data.where((data['income']==' >50K')).where(data[field]==value).count()/data.count()).iloc[0]
	pn50=1-p50
	pn50k=p50-p50k
	return plogp(p50k)+plogp(pn50k), p50k*np.log(p50k/p50)+pn50k*np.log(pn50k/pn50)

def plogp(p): return p*np.log(p)

df = pd.read_csv('data/adult.data',header=None,names=['age', 'workclass', 'fnlwgt', 'education','education-num', 
		'marital-status','occupation', 'relationship', 'race','sex','capital-gain', 'capital-loss',
       	'hours-per-week', 'country','income'])
#get_frequencies(df)

def check_new(fname):
	data= pd.read_csv(fname)
	conversion = data.set_index('Value')['New'].to_dict()
	new = data.set_index('Value').groupby('New').apply(lambda x: (np.sum(x['Frequency']*x['Probability']))/np.sum(x['Frequency']))
	old = data.set_index('Value')['Probability']
	diff=pd.Series([old[v]-new[conversion[v]] for v in old.index],old.index)
	return new,old,diff
