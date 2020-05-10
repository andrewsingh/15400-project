import sys
import numpy as np
import random
import pandas as pd
import pickle
import time
import argparse
import alt_min
from os import path



trial_num = 1



def average_attack(ModelClass, reg, verbose):
  train = pd.read_pickle("../data/ml-1m-split/train.pkl").drop(["item_id", "timestamp"], axis=1)
  test = pd.read_pickle("../data/ml-1m-split/test.pkl").drop(["item_id", "timestamp"], axis=1)
  full = pd.read_pickle("../data/ml-1m-split/full.pkl").drop(["item_id", "timestamp"], axis=1)

  num_users = len(full.groupby("user").size())
  num_items = len(full.groupby("item").size())
  num_factors = 30

  item_freqs = full.groupby("item").size().values
  item_freqs_train = train.groupby("item").size()
  item_rating_avgs = full.groupby("item").mean()["rating"].values
  item_rating_stds = full.groupby("item").std()["rating"].values

  # target_items_list = []
  # for i in range(num_items):
  #   if i in item_freqs_train and item_freqs[i] <= (0.05 * num_users) and item_freqs[i] >= (0.03 * num_users) and item_rating_avgs[i] >= 3:
  #     target_items_list.append(i)

  # target_items = random.sample(target_items_list, k=10)

  # target_items = [117, 1792, 2837, 3157, 2206, 3038, 1597, 3466, 1988, 3014]
  # target_items = [371, 1100, 2531, 40, 2818, 1314, 1747, 3081, 2984, 871]
  #target_items = [371, 1100, 2531, 40, 2818]
  target_items = [40, 2818]
  # print("Target items: {}\n".format(target_items))

  # filler_prop = 0.05
  # filler_size = int(filler_prop * num_items)
  # filler_items_list = [x for x in list(range(num_items)) if x in item_freqs_train and x not in target_items]

  # model_clean = ModelClass(train, test, num_items, num_factors=num_factors, reg=reg)
  # model_clean.alt_min()
  # overall_rmse = model_clean.evaluate()
  # clean_results = []
  results = []
  
  # for target_item in target_items:
  #   target_rmse = model_clean.evaluate_item(target_item)
  #   original_entry = [target_item, 0.0, 0.0, round(target_rmse, 4), round(overall_rmse, 4)]
  #   clean_results.append(original_entry)
  #   print(original_entry)

  # with open("../results/drop_attack/clean{}.pkl".format(alt_min.ModelClass.__name__), "wb+") as f:
  #     pickle.dump(clean_results, f)
    
  
  for target_item in target_items:
    if verbose:
      print("Target item {} freq: {}".format(target_item, len(train.loc[train["item"] == target_item])))

    #np.random.seed(0)
    # filler_items = random.sample(filler_items_list, k=filler_size)
    # target_rmse = model_clean.evaluate_item(target_item)

    # original_entry = [target_item, 0.0, 0.0, round(target_rmse, 4), round(overall_rmse, 4)]
    # results.append(original_entry)
    # print(original_entry)
    
    # for eta in [0.01, 0.03, 0.05, 0.10, 0.25]:
    for eta in [0.1, 0.2]:
      # profile_size = int(profile_prop * num_users)
      profile_size = int(item_freqs_train[target_item] * eta)
      attack_data = []
      for i in range(profile_size):
        user = i + num_users
        # Drop target item
        attack_data.append([user, target_item, -100])

        # for filler_item in filler_items:
        #   if np.isnan(item_rating_stds[filler_item]):
        #     raw_rating = item_rating_avgs[filler_item]
        #   else:
        #     raw_rating = np.random.normal(item_rating_avgs[filler_item], item_rating_stds[filler_item])
          
        #   rating = int(round(np.clip(raw_rating, 1, 5)))
        #   attack_data.append([user, filler_item, rating])

      attack_df = pd.DataFrame(attack_data).rename(columns={0: "user", 1: "item", 2: "rating"})
      train_attacked = train.append(attack_df).reset_index().drop(["index"], axis=1)
      if ModelClass.__name__ == "HuberGradient":
        model_attack = ModelClass(train_attacked, test, num_items, num_factors=num_factors, reg=reg, corruption=eta)
      else: 
        model_attack = ModelClass(train_attacked, test, num_items, num_factors=num_factors, reg=reg)
      model_attack.alt_min()
      overall_rmse_attacked = model_attack.evaluate()
      target_rmse_attacked = model_attack.evaluate_item(target_item)
      
      attack_prop = profile_size / (profile_size + len(train.loc[train["item"] == target_item]))
      entry = [target_item, eta, round(attack_prop, 4), \
              round(target_rmse_attacked, 4), round(overall_rmse_attacked, 4)]
      results.append(entry)
      print(entry)

      with open("../results/drop_attack/{}_{}.pkl".format(ModelClass.__name__, eta), "wb+") as f:
        pickle.dump(results, f)

  # with open("trial1_LeastSquares.pkl", "rb") as f:
  #   (target_items, results) = pickle.load(f)




def corrupt_data(data, num_users, num_items, eta, c, b):
  if eta == 0:
    return data
  data_path = "../data/ml-1m-noisy/train{}_{}_{}.pkl".format(eta, c, b)
  if path.exists(data_path):
    return pd.read_pickle(data_path)

  data_u = data.sort_values(by="user", axis=0).reset_index(drop=True).to_numpy()
  data_v = data.sort_values(by="item", axis=0).reset_index(drop=True).to_numpy()
  len_data = len(data)

  U_freqs = data.groupby("user").size().values
  V_group = data.groupby("item").size()
  V_index = V_group.index.values
  V_freqs = np.zeros(num_items, dtype="int")
  for i in V_index:
    V_freqs[i] = V_group[i]

  u_corruptions = np.zeros(num_users).astype(int)
  v_corruptions = np.zeros(num_items).astype(int)
  corruptions = 0

  data = data.sample(frac=1).reset_index(drop=True)

  for (index, [user, item, rating]) in data.iterrows():
    user_frac = u_corruptions[user] / U_freqs[user]
    item_frac = v_corruptions[item] / V_freqs[item]
    if user_frac < eta and item_frac < eta:
      data.loc[index][2] = np.random.uniform(-c + b, c + b)
      corruptions += 1
      u_corruptions[user] += 1
      v_corruptions[item] += 1
    if corruptions == int(len_data * eta):
      break
    if index % 10000 == 0:
      print(index)
      print("corruption fraction: {}\n".format(corruptions / len_data))

  data.to_pickle("../data/ml-1m-noisy/train{}_{}_{}.pkl".format(eta, c, b))
  return data





def run_experiment(ModelClass, reg, r, eta, c, b, verbose):
  train = pd.read_pickle("../data/ml-1m-split/train.pkl").drop(["item_id", "timestamp"], axis=1)
  test = pd.read_pickle("../data/ml-1m-split/test.pkl").drop(["item_id", "timestamp"], axis=1)
  full = pd.read_pickle("../data/ml-1m-split/full.pkl").drop(["item_id", "timestamp"], axis=1)

  num_users = len(full.groupby("user").size())
  num_items = len(full.groupby("item").size())

  train = corrupt_data(train, num_users, num_items, eta, c, b)

  if ModelClass.__name__ in ["HuberGradient", "RemoveOutliers"]:
    model = ModelClass(train, test, num_items, num_factors=r, reg=reg, corruption=eta)
  else:
    model = ModelClass(train, test, num_items, num_factors=r, reg=reg)
  
  return model.alt_min(verbose)




if __name__ == '__main__':
  np.random.seed(0)
  
  parser = argparse.ArgumentParser(description='Alt-Min matrix completion algorithm')

  parser.add_argument("-r", dest="r", type=int)
  parser.add_argument("-eta", dest="eta", type=float)
  parser.add_argument("-c", dest="c", type=float, default=50)
  parser.add_argument("-b", dest="b", type=float, default=0)
  parser.add_argument("-m", dest="mclass")
  parser.add_argument("-reg", dest="reg", type=float)
  parser.add_argument("-v", action="store_true")
  parser.add_argument("-aa", action="store_true")
  # parser.add_argument("-t", dest="target_item")
  
  args = parser.parse_args() 

  if args.mclass == "ls":
    print("Training least squares model")
    ModelClass = alt_min.LeastSquares
  elif args.mclass == "lad":
    print("Training least abs dev model")
    ModelClass = alt_min.LeastAbsDev
  elif args.mclass == "hg":
    print("Training Huber gradient model")
    ModelClass = alt_min.HuberGradient
  else:
    ModelClass = alt_min.LeastSquares

  if args.aa:
    average_attack(ModelClass, args.reg, args.v)
  else:
    rmse = run_experiment(ModelClass, args.reg, args.r, args.eta, args.c, args.b, args.v)
    print("Experiment complete\nargs: {}".format(args))
    with open("../results/movielens/ml_exp_log.txt", "a+") as f:
      f.write("{} final RMSE: {}\n".format(args, rmse))