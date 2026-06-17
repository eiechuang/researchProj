import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd


filepath = '/Users/erichuang/Documents/dev/Python/'
filename = 'clients_with_fatf_ofac.csv'

names = ('client-id','client-name','client-type','sector',"sector-risk", 'country', 'PEP', 'Sanctioned', 'FATF', 'OFTF')
df = pd.read.csv (filepath+filename)

df.drop('client-id','client-name')

fatf-grey = 0.5
fatf-black = 1.5

final_country_risk = countryRisk + 
model = XGBClassifier


