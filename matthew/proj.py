from __future__ import print_function
import ast
import csv
import json
import numpy as np
import pandas as pd
import time as tm
import datetime as dt
import itertools as itt
from surprise import Dataset
from surprise import accuracy
from surprise import Reader
from surprise.prediction_algorithms.knns import KNNBasic
from surprise.prediction_algorithms.knns import KNNWithMeans
from surprise.prediction_algorithms.knns import KNNWithZScore
from surprise.prediction_algorithms.knns import KNNBaseline
from surprise.prediction_algorithms.matrix_factorization import SVD
from surprise.prediction_algorithms.matrix_factorization import SVDpp
from surprise.prediction_algorithms.matrix_factorization import NMF
from surprise.prediction_algorithms.slope_one import SlopeOne
from surprise.prediction_algorithms.co_clustering import CoClustering
from surprise.model_selection import train_test_split
from surprise.model_selection import cross_validate
from surprise.model_selection import KFold
from surprise.model_selection.search import GridSearchCV

def main():
  rec()

def rec():
  reviewsPath = 'data/reviews_ssc.csv'
  df_reviews = pd.read_csv(reviewsPath, sep=',')
  df_reviews['unixReviewTime'] = pd.to_numeric(df_reviews['unixReviewTime'], errors='coerce')

  reader = Reader(line_format='user item rating timestamp', sep=',', rating_scale=(1,5), skip_lines=1)
  reviewsData = Dataset.load_from_file(reviewsPath, reader=reader)
  trainset, testset = train_test_split(reviewsData, test_size=.25)

  """
  param_grid = {'k':[40,50],
                'min_k':[3,7],
                'sim_options': {'name': ['msd'],
                                'min_support': [1,5],
                                'user_based': [False]}}
  gs = GridSearchCV(KNNWithMeans, param_grid, measures=['rmse'],cv=5)
  gs.fit(reviewsData)
  print(gs.best_score['rmse'])
  print(gs.best_params['rmse'])"""

  results = []
  n_cltr_u = [3,5,7,9,11]
  n_cltr_i = [3,5,7,9,11]
  for a in n_cltr_u:
    for b in n_cltr_i:
      algo = CoClustering(n_cltr_u=a,n_cltr_i=b)
      predictions = algo.fit(trainset).test(testset)
      rmse = accuracy.rmse(predictions, verbose=False)
      mae = accuracy.mae(predictions, verbose=False)
      fcp = accuracy.fcp(predictions, verbose=False)
      results.append((rmse,mae,fcp,a,b))
      print('{} {} {} {} {}'.format(rmse,mae,fcp,a,b))

  #rows = sorted(results, key=lambda x: x[0])
  df = pd.DataFrame(results, columns=['rmse','mae','fcp','k','min_k'])
  df.to_csv('co_clustering.csv',index=False)

  """
    param_grid = {'lr_pu': [0.019775, 0.019825],
                'reg_bi': [0.06275, 0.06325],
                'reg_pu': [0.20775, 0.20825],
                'lr_bu': [0.01075, 0.01125],
                'lr_bi': [0.005275, 0.005325],
                'reg_bu': [0.06675, 0.06725],
                'reg_qi': [0.14775, 0.14825],
                'lr_qi': [0.014775, 0.014825]}
  results = []
  lr_bu = [0.001,0.005,0.01]
  lr_bi = [0.001,0.005,0.01]
  lr_pu = [0.001,0.005,0.01]
  lr_qi = [0.001,0.005,0.01]
  reg_bu = [0.005,0.02,0.05]
  reg_bi = [0.005,0.02,0.05]
  reg_pu = [0.005,0.02,0.05]
  reg_qi = [0.005,0.02,0.05]
  g = itt.product(lr_bu,lr_bi,lr_pu,lr_qi,reg_bu,reg_bi,reg_pu,reg_qi)
  for i in g:
    algo = SVD(n_factors=200,n_epochs=50,lr_bu=i[0],lr_bi=i[1],lr_pu=i[2],
               lr_qi=i[3],reg_bu=i[4],reg_bi=i[5],reg_pu=i[6],reg_qi=i[7])
    predictions = algo.fit(trainset).test(testset)
    acc = accuracy.rmse(predictions, verbose=False)
    results.append((acc,)+i)

  rows = sorted(results, key=lambda x: x[0])
  df = pd.DataFrame(rows, columns=['rmse','lr_bu','lr_bi','lr_pu','lr_qi',
                                   'reg_bu','reg_bi','reg_pu','reg_qi'])
  df.to_csv('svd.csv',index=False)"""

  print('done')

def placesJsonToCsv():
  cnt = 0

  with open('places.csv', 'w', encoding='utf-8', newline='') as c:
    w = csv.writer(c, delimiter='\t')
    w.writerow(['name','address','gPlusPlaceId','latitude','longitude'])

    with open('data/places.json', 'r', encoding='utf-8', newline='') as f:
      for line in f:
        j = ast.literal_eval(line)
        
        name = j['name']
        if isinstance(name, bytes):
          name = str(name, 'utf-8')
        address = ','.join(j['address'])
        if isinstance(address, bytes):
          address = str(address, 'utf-8')
        gPlusPlaceId = j['gPlusPlaceId']
        gps = j['gps']
        latitude = gps[0] if gps else None
        longitude = gps[1] if gps else None

        w.writerow([name.encode('utf-8'),address.encode('utf-8'),gPlusPlaceId,latitude,longitude])

        cnt += 1
        if cnt % 1000 == 0:
          print(cnt)

if __name__ == '__main__':
  main()

























