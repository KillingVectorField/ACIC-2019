import numpy as np
import pandas as pd

seed=1

data=pd.read_csv(r"ACIC 2019\TestDatasets_lowD\testdataset2.csv")
n,p=data.shape

from sklearn.ensemble import RandomForestRegressor

regr=RandomForestRegressor(random_state=seed).fit(data[data.columns.drop('Y')],data['Y'])
pred_treat=regr.predict(np.column_stack(([1]*n,data[data.columns.drop(['Y','A'])])))
pred_control=regr.predict(np.column_stack(([0]*n,data[data.columns.drop(['Y','A'])])))
reg_ATE=np.mean(pred_treat-pred_control)
print(reg_ATE)

from sklearn.linear_model import LogisticRegression
clf=LogisticRegression(random_state=seed).fit(data[data.columns.drop(['Y','A'])],data['A'])
propensity=clf.predict_proba(data[data.columns.drop(['Y','A'])])[:,1]
ipw_ATE=np.mean(data['Y']/propensity*data['A']-data['Y']/(1-propensity)*(1-data['A']))
print(ipw_ATE)

# double robust (AIPW)
mu_1=np.mean(pred_treat+data['A']*(data['Y']-regr.predict(data[data.columns.drop('Y')]))/propensity)
mu_0=np.mean(pred_control+(1-data['A'])*(data['Y']-regr.predict(data[data.columns.drop('Y')]))/(1-propensity))
aipw_ATE=mu_1-mu_0