import sys
import numpy as np
import random
import pandas as pd
import pickle
import time


verbose = True


class MatrixFactorization:
  def __init__(self, train, test, num_items, num_factors, lrate, reg):
    self.num_train_users = len(train.groupby("user").size())
    self.num_test_users = len(test.groupby("user").size())
    self.num_items = num_items
    self.num_factors = num_factors
    self.lrate = lrate
    self.reg = reg

    self.train_u = train.sort_values(by="user", axis=0).reset_index(drop=True).to_numpy()
    self.train_v = train.sort_values(by="item", axis=0).reset_index(drop=True).to_numpy()
    self.test_u = test.sort_values(by="user", axis=0).reset_index(drop=True).to_numpy()
    self.test_v = test.sort_values(by="item", axis=0).reset_index(drop=True).to_numpy()

    U_freqs = train.groupby("user").size().values
    U_freqs_test = test.groupby("user").size().values

    V_group = train.groupby("item").size()
    self.V_index = V_group.index.values
    V_freqs = np.zeros(num_items, dtype="int")
    for i in self.V_index:
      V_freqs[i] = V_group[i]

    V_group_test = test.groupby("item").size()
    V_index_test = V_group_test.index.values
    V_freqs_test = np.zeros(num_items, dtype="int")
    for i in V_index_test:
      V_freqs_test[i] = V_group_test[i]
      
    self.U_start = np.insert(np.cumsum(U_freqs), 0, 0)
    self.U_start_test = np.insert(np.cumsum(U_freqs_test), 0, 0)
    self.V_start = np.insert(np.cumsum(V_freqs), 0, 0)
    self.V_start_test = np.insert(np.cumsum(V_freqs_test), 0, 0)



  def evaluate(self, test=True):
    if test:
      data = self.test_u
      start = self.U_start_test
      num_users = self.num_test_users
    else:
      data = self.train_u
      start = self.U_start
      num_users = self.num_train_users

    square_error = 0
    for user in range(num_users):
      user_data = data[start[user] : start[user + 1]]
      square_error += np.sum((user_data[:, 2] - np.matmul(self.V[user_data[:, 1]], self.U[user])) ** 2)
    return np.sqrt(square_error / len(data))


  def evaluate_item(self, item):
    data = self.test_v[self.V_start_test[item] : self.V_start_test[item + 1]]
    return np.sqrt(np.sum((data[:, 2] - np.matmul(self.U[data[:, 0]], self.V[item])) ** 2) / len(data))


  def get_u_step(self, user):
    pass


  def get_v_step(self, item):
    pass


  def alt_min(self):
    self.U = np.random.uniform(-1, 1, (self.num_train_users, self.num_factors))
    self.V = np.random.uniform(-1, 1, (self.num_items, self.num_factors))

    rmse = self.evaluate()
    prev_rmse = 1000
    rounds = 0
    num_iters = 20
    threshold = -0.001

    while rmse - prev_rmse < threshold:
      if verbose:
        t0 = time.time()
      prev_rmse = rmse
      
      # Optmize U
      for i in range(num_iters):
        for user in range(self.num_train_users):
          step = self.get_u_step(user)
          self.U[user] += self.lrate * step
          
      # Optimize V
      for i in range(num_iters):
        for item in self.V_index:
          step = self.get_v_step(item)
          self.V[item] += self.lrate * step
          
      rmse = self.evaluate()
      rounds += 1

      if verbose:
        t1 = time.time()
        train_rmse = self.evaluate(False)
        print("\n==================== ROUND {} ====================\nRMSE: {}\nPrev RMSE: {}\nDiff: {}\nTrain RMSE: {}\nExecution time: {}\n" \
          .format(rounds, round(rmse, 4), round(prev_rmse, 4), round(rmse - prev_rmse, 4), round(train_rmse, 4), round(t1 - t0, 2)))
      
    if verbose:
      print("max U: {}\nmin U: {}\navg U: {}\n".format(np.amax(self.U), np.amin(self.U), np.mean(self.U)))
      print("max V: {}\nmin V: {}\navg V: {}\n".format(np.amax(self.V), np.amin(self.V), np.mean(self.V)))
      print("Final RMSE: {}\n".format(rmse))




class LeastSquares(MatrixFactorization):
  def __init__(self, train, test, num_items, num_factors=30, lrate=0.1, reg=0.05):
    MatrixFactorization.__init__(self, train, test, num_items, num_factors, lrate, reg)


  def get_u_step(self, user):
    data = self.train_u[self.U_start[user] : self.U_start[user + 1]]
    vmat = self.V[data[:, 1]]
    preds = np.matmul(vmat, self.U[user])
    return np.mean(np.multiply((data[:, 2] - preds).reshape((-1, 1)), vmat) - (self.reg * self.U[user]), axis=0)

    
  def get_v_step(self, item):
    data = self.train_v[self.V_start[item] : self.V_start[item + 1]]
    umat = self.U[data[:, 0]]
    preds = np.matmul(umat, self.V[item])
    return np.mean(np.multiply((data[:, 2] - preds).reshape((-1, 1)), umat) - (self.reg * self.V[item]), axis=0)



class LeastAbsDev(MatrixFactorization):
  def __init__(self, train, test, num_items, num_factors=30, lrate=0.1, reg=0.05):
    MatrixFactorization.__init__(self, train, test, num_items, num_factors, lrate, reg)


  def get_u_step(self, user):
    data = self.train_u[self.U_start[user] : self.U_start[user + 1]]
    vmat = self.V[data[:, 1]]
    preds = np.matmul(vmat, self.U[user])
    return np.mean(np.multiply(np.sign(data[:, 2] - preds).reshape((-1, 1)), vmat) - (self.reg * self.U[user]), axis=0)

    
  def get_v_step(self, item):
    data = self.train_v[self.V_start[item] : self.V_start[item + 1]]
    umat = self.U[data[:, 0]]
    preds = np.matmul(umat, self.V[item])
    return np.mean(np.multiply(np.sign((data[:, 2] - preds)).reshape((-1, 1)), umat) - (self.reg * self.V[item]), axis=0)



class MedianGradient(MatrixFactorization):
  def __init__(self, train, test, num_items, num_factors=30, lrate=0.1, reg=0.05):
    MatrixFactorization.__init__(self, train, test, num_items, num_factors, lrate, reg)


  def get_u_step(self, user):
    data = self.train_u[self.U_start[user] : self.U_start[user + 1]]
    vmat = self.V[data[:, 1]]
    preds = np.matmul(vmat, self.U[user])
    return np.median(np.multiply((data[:, 2] - preds).reshape((-1, 1)), vmat) - (self.reg * self.U[user]), axis=0)

    
  def get_v_step(self, item):
    data = self.train_v[self.V_start[item] : self.V_start[item + 1]]
    umat = self.U[data[:, 0]]
    preds = np.matmul(umat, self.V[item])
    return np.median(np.multiply((data[:, 2] - preds).reshape((-1, 1)), umat) - (self.reg * self.V[item]), axis=0)



class HuberLossGradient(MatrixFactorization):
  def __init__(self, train, test, num_items, num_factors=30, lrate=0.1, reg=0.05, delta=1):
    MatrixFactorization.__init__(self, train, test, num_items, num_factors, lrate, reg)
    self.delta = delta

  def get_u_step(self, user):
    data = self.train_u[self.U_start[user] : self.U_start[user + 1]]
    vmat = self.V[data[:, 1]]
    preds = np.matmul(vmat, self.U[user])
    residuals = data[:, 2] - preds
    # print("User {}\nmean: {}\nstd: {}\n".format(user, np.mean(residuals), np.std(residuals)))
    dl_dres = np.where(residuals <= self.delta, residuals, self.delta * np.sign(residuals))
    return np.mean(np.multiply(dl_dres.reshape((-1, 1)), vmat) - (self.reg * self.U[user]), axis=0)

    
  def get_v_step(self, item):
    data = self.train_v[self.V_start[item] : self.V_start[item + 1]]
    umat = self.U[data[:, 0]]
    preds = np.matmul(umat, self.V[item])
    residuals = data[:, 2] - preds
    dl_dres = np.where(residuals <= self.delta, residuals, self.delta * np.sign(residuals))
    return np.mean(np.multiply(dl_dres.reshape((-1, 1)), umat) - (self.reg * self.V[item]), axis=0)




def huberGradientEst(grads):
  grad_cov = np.cov(grads)
  (u, s, vh) = np.linalg.svd(grad_cov)


class HuberGradient(MatrixFactorization):
  def __init__(self, train, test, num_items, num_factors=30, lrate=0.1, reg=0.05):
    MatrixFactorization.__init__(self, train, test, num_items, num_factors, lrate, reg)


  def get_u_step(self, user):
    data = self.train_u[self.U_start[user] : self.U_start[user + 1]]
    vmat = self.V[data[:, 1]]
    preds = np.matmul(vmat, self.U[user])
    return huberGradientEst(np.multiply((data[:, 2] - preds).reshape((-1, 1)), vmat) - (self.reg * self.U[user]))

    
  def get_v_step(self, item):
    data = self.train_v[self.V_start[item] : self.V_start[item + 1]]
    umat = self.U[data[:, 0]]
    preds = np.matmul(umat, self.V[item])
    return huberGradientEst(np.multiply((data[:, 2] - preds).reshape((-1, 1)), umat) - (self.reg * self.V[item]))





def average_attack(ModelClass):
  train = pd.read_pickle("../data/ml-1m-split/train.pkl").drop(["item_id", "timestamp"], axis=1)
  test = pd.read_pickle("../data/ml-1m-split/test.pkl").drop(["item_id", "timestamp"], axis=1)
  full = pd.read_pickle("../data/ml-1m-split/full.pkl").drop(["item_id", "timestamp"], axis=1)

  num_users = len(full.groupby("user").size())
  num_items = len(full.groupby("item").size())

  model_clean = ModelClass(train, test, num_items)
  
  item_freqs = full.groupby("item").size().values
  item_freqs_train = train.groupby("item").size()
  item_rating_avgs = full.groupby("item").mean()["rating"].values
  item_rating_stds = full.groupby("item").std()["rating"].values

  target_items_list = []
  for i in range(num_items):
    if i in item_freqs_train and item_freqs[i] <= (0.05 * num_users) and item_freqs[i] >= (0.02 * num_users) and item_rating_avgs[i] < 3:
      target_items_list.append(i)

  filler_prop = 0.05
  filler_size = int(filler_prop * num_items)
  filler_items_list = list(range(num_items))
  filler_items_list = [x for x in filler_items_list if x in item_freqs_train]


  model_clean.alt_min()
  overall_rmse = model_clean.evaluate()
  results = []
  # target_items = random.sample(target_items_list, k=5)
  target_items = [2582, 2611, 2186, 3643, 2123]
  print("Target items: {}\n".format(target_items))
  
  for target_item in target_items:
    if verbose:
      print("Target item {} freq: {}".format(target_item, len(train.loc[train["item"] == target_item])))
    filler_items = random.sample([x for x in filler_items_list if x != target_item], k=filler_size)
    target_rmse = model_clean.evaluate_item(target_item)
    
    original_entry = [target_item, 0.0, round(target_rmse, 4), round(overall_rmse, 4)]
    results.append(original_entry)
    print(original_entry)
    
    for profile_prop in [0.01, 0.05, 0.10]:
      profile_size = int(profile_prop * num_users)
      attack_data = []
      for i in range(profile_size):
        user = i + num_users
        # Boost target item
        attack_data.append([user, target_item, 5]) 

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
      entry = [target_item, round(attack_prop, 4), \
              round(target_rmse_attacked, 4), round(overall_rmse_attacked, 4)]
      results.append(entry)
      print(entry)

  with open("../results/average_attack/trial1/trial1_{}.pkl".format(ModelClass.__name__)) as f:
    pickle.dump((target_items, results), f)




if __name__ == '__main__':
  args = sys.argv
  if len(args) >= 2:
    if args[1] != "attack":
      train = pd.read_pickle("../data/ml-1m-split/train.pkl").drop(["item_id", "timestamp"], axis=1)
      test = pd.read_pickle("../data/ml-1m-split/test.pkl").drop(["item_id", "timestamp"], axis=1)
      full = pd.read_pickle("../data/ml-1m-split/full.pkl").drop(["item_id", "timestamp"], axis=1)

      num_users = len(full.groupby("user").size())
      num_items = len(full.groupby("item").size())

      if len(args) == 3:
        reg = float(args[2])
      else:
        reg = 0.05

      if args[1] == "ls":
        print("Training least squares model with reg = {}".format(reg))
        ls = LeastSquares(train, test, num_items, reg=reg)
        ls.alt_min()
      elif args[1] == "lad":
        print("Training least abs dev model with reg = {}".format(reg))
        lad = LeastAbsDev(train, test, num_items, reg=reg)
        lad.alt_min()
      elif args[1] == "mgrad":
        print("Training median gradient model with reg = {}".format(reg))
        mgrad = MedianGradient(train, test, num_items, reg=reg)
        mgrad.alt_min()
      elif args[1] == "hlgrad":
        print("Training Huber loss gradient model with reg = {}".format(reg))
        hlgrad = HuberLossGradient(train, test, num_items, reg=reg)
        hlgrad.alt_min()
      
    elif args[1] == "attack" and len(args) == 3:
      if args[2] == "ls":
        print("Attacking least squares model")
        average_attack(LeastSquares)
      elif args[2] == "lad":
        print("Attacking least abs dev model")
        average_attack(LeastAbsDev)
      elif args[2] == "mgrad":
        print("Attacking median gradient model")
        average_attack(MedianGradient)
      elif args[2] == "hlgrad":
        print("Attacking Huber loss gradient model")
        average_attack(HuberLossGradient)
      elif args[2] == "all":
        print("Attacking all models")
        average_attack(LeastSquares)
        average_attack(LeastAbsDev)
        average_attack(MedianGradient)
        average_attack(HuberLossGradient)
      

    



    
    