import sys
import numpy as np
import random
import pandas as pd
import pickle
import time
import alt_min


trial_num = 2

def average_attack(ModelClass):
  train = pd.read_pickle("../data/ml-1m-split/train.pkl").drop(["item_id", "timestamp"], axis=1)
  test = pd.read_pickle("../data/ml-1m-split/test.pkl").drop(["item_id", "timestamp"], axis=1)
  full = pd.read_pickle("../data/ml-1m-split/full.pkl").drop(["item_id", "timestamp"], axis=1)

  num_users = len(full.groupby("user").size())
  num_items = len(full.groupby("item").size())

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
  target_items = [371, 1100, 2531, 40, 2818, 1314, 1747, 3081, 2984, 871]

  print("Target items: {}\n".format(target_items))

  filler_prop = 0.05
  filler_size = int(filler_prop * num_items)
  filler_items_list = [x for x in list(range(num_items)) if x in item_freqs_train and x not in target_items]

  model_clean = ModelClass(train, test, num_items)
  model_clean.alt_min()
  overall_rmse = model_clean.evaluate()
  results = []
  
  for target_item in target_items:
    if verbose:
      print("Target item {} freq: {}".format(target_item, len(train.loc[train["item"] == target_item])))

    np.random.seed(0)
    filler_items = random.sample(filler_items_list, k=filler_size)
    target_rmse = model_clean.evaluate_item(target_item)

    original_entry = [target_item, 0.0, 0.0, round(target_rmse, 4), round(overall_rmse, 4)]
    results.append(original_entry)
    print(original_entry)
    
    for profile_prop in [0.01, 0.03, 0.05, 0.10, 0.25]:
      profile_size = int(profile_prop * num_users)
      attack_data = []
      for i in range(profile_size):
        user = i + num_users
        # Drop target item
        attack_data.append([user, target_item, 1])

        for filler_item in filler_items:
          if np.isnan(item_rating_stds[filler_item]):
            raw_rating = item_rating_avgs[filler_item]
          else:
            raw_rating = np.random.normal(item_rating_avgs[filler_item], item_rating_stds[filler_item])
          
          rating = int(round(np.clip(raw_rating, 1, 5)))
          attack_data.append([user, filler_item, rating])

      attack_df = pd.DataFrame(attack_data).rename(columns={0: "user", 1: "item", 2: "rating"})
      train_attacked = train.append(attack_df).reset_index().drop(["index"], axis=1)
      model_attack = ModelClass(train_attacked, test, num_items)
      model_attack.alt_min()
      overall_rmse_attacked = model_attack.evaluate()
      target_rmse_attacked = model_attack.evaluate_item(target_item)
      
      attack_prop = profile_size / (profile_size + len(train.loc[train["item"] == target_item]))
      entry = [target_item, profile_prop, round(attack_prop, 4), \
              round(target_rmse_attacked, 4), round(overall_rmse_attacked, 4)]
      results.append(entry)
      print(entry)
     

  with open("../results/average_attack/trial{}/trial{}_{}.pkl".format(trial_num, trial_num, alt_min.ModelClass.__name__), "wb+") as f:
    pickle.dump((target_items, results), f)

  # with open("trial1_LeastSquares.pkl", "rb") as f:
  #   (target_items, results) = pickle.load(f)





if __name__ == '__main__':
  args = sys.argv
  if len(args) >= 2:
    if args[1] != "attack":
      verbose = True
      train = pd.read_pickle("../data/ml-1m-split/train.pkl").drop(["item_id", "timestamp"], axis=1)
      test = pd.read_pickle("../data/ml-1m-split/test.pkl").drop(["item_id", "timestamp"], axis=1)
      full = pd.read_pickle("../data/ml-1m-split/full.pkl").drop(["item_id", "timestamp"], axis=1)

      num_users = len(full.groupby("user").size())
      num_items = len(full.groupby("item").size())

      if args[1] == "ls":
        print("Training least squares model")
        ModelClass = alt_min.LeastSquares
      elif args[1] == "lad":
        print("Training least abs dev model")
        ModelClass = alt_min.LeastAbsDev
      elif args[1] == "hl":
        print("Training Huber loss model")
        ModelClass = alt_min.HuberLoss
      elif args[1] == "hg":
        print("Training Huber gradient model")
        ModelClass = alt_min.HuberGradient

      if len(args) == 4:
        if args[2] == "reg":
          model = ModelClass(train, test, num_items, reg=float(args[3]))
          print("Regularization: {}".format(args[3]))
        elif args[2] == "delta":
          model = ModelClass(train, test, num_items, delta=float(args[3]))
          print("Delta: {}".format(args[3]))
        else:
          model = ModelClass(train, test, num_items)
      elif len(args) == 5 and args[2] == "reg" and args[3] == "delta":
        model = ModelClass(train, test, num_items, reg=float(args[3]), delta=float(args[4]))
        print("Regularization: {}\nDelta: {}".format(args[3], args[4]))
      else:
        model = ModelClass(train, test, num_items)
        
      model.alt_min()
    

    elif args[1] == "attack" and len(args) == 3:
      if args[2] == "ls":
        print("Attacking least squares model")
        average_attack(alt_min.LeastSquares)
      elif args[2] == "lad":
        print("Attacking least abs dev model")
        average_attack(alt_min.LeastAbsDev)
      elif args[2] == "hl":
        print("Attacking Huber loss model")
        average_attack(alt_min.HuberLoss)

  else:
    X = np.random.standard_normal((100, 30))
    X[range(0, 100, 10)] = np.random.uniform(-10, 10, (10, 30))
    est = alt_min.agnosticMeanGeneral(X, 0.1)
    print("mean norm: {}".format(np.linalg.norm(np.mean(X, axis=0))))
    print("median norm: {}".format(np.linalg.norm(np.median(X, axis=0))))
    print("est norm: {}".format(np.linalg.norm(est)))