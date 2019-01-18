import numpy as np
import pandas as pd

randomseednumber = 1
np.random.seed(randomseednumber)

data=pd.read_csv(r"ACIC 2019\TestDatasets_lowD\testdataset2.csv")
n,p=data.shape

from sklearn.ensemble import RandomForestRegressor

regr=RandomForestRegressor(max_depth=3).fit(data[data.columns.drop('Y')],data['Y'])
pred_treat=regr.predict(np.column_stack(([1]*n,data[data.columns.drop(['Y','A'])])))
pred_control=regr.predict(np.column_stack(([0]*n,data[data.columns.drop(['Y','A'])])))
reg_ATE=np.mean(pred_treat-pred_control)
print(reg_ATE)

from sklearn.linear_model import LogisticRegression
clf=LogisticRegression().fit(data[data.columns.drop(['Y','A'])],data['A'])
propensity=clf.predict_proba(data[data.columns.drop(['Y','A'])])[:,1]
ipw_ATE=np.mean(data['Y']/propensity*data['A']-data['Y']/(1-propensity)*(1-data['A']))
print(ipw_ATE)

# double robust (AIPW)
mu_1=np.mean(pred_treat+data['A']*(data['Y']-regr.predict(data[data.columns.drop('Y')]))/propensity)
mu_0=np.mean(pred_control+(1-data['A'])*(data['Y']-regr.predict(data[data.columns.drop('Y')]))/(1-propensity))
aipw_ATE=mu_1-mu_0

# double machine learning (partial linear model)
I = np.random.choice(n,np.int(n/2),replace=False)
I_C = [x for x in np.arange(n) if x not in I]

Ghat_1 = RandomForestRegressor().fit(data[data.columns.drop(['Y','A'])].loc[I],data['Y'].loc[I]).predict(data[data.columns.drop(['Y','A'])].loc[I_C])
Ghat_2 = RandomForestRegressor().fit(data[data.columns.drop(['Y','A'])].loc[I_C],data['Y'].loc[I_C]).predict(data[data.columns.drop(['Y','A'])].loc[I])
    
Mhat_1 = RandomForestRegressor().fit(data[data.columns.drop(['Y','A'])].loc[I],data['A'].loc[I]).predict(data[data.columns.drop(['Y','A'])].loc[I_C])
Mhat_2 = RandomForestRegressor().fit(data[data.columns.drop(['Y','A'])].loc[I_C],data['A'].loc[I_C]).predict(data[data.columns.drop(['Y','A'])].loc[I])

Vhat_1 = data['A'][I_C]-Mhat_1
Vhat_2 = data['A'][I] - Mhat_2

theta_1 = np.mean(np.dot(Vhat_1,(data['Y'].loc[I_C]-Ghat_1)))/np.mean(np.dot(Vhat_1,data['A'].loc[I_C]))
theta_2 = np.mean(np.dot(Vhat_2,(data['Y'].loc[I]-Ghat_2)))/np.mean(np.dot(Vhat_2,data['A'].loc[I]))
DML_plm_ATE = 0.5*(theta_1+theta_2)


