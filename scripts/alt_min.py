import sys
import numpy as np
import random
import pandas as pd
import pickle
import time
from lenskit.algorithms.als import BiasedMF
from lenskit.batch import predict
from lenskit.metrics.predict import rmse


verbose = False
trial_num = 2


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
    np.random.seed(0)
    self.U = np.random.uniform(-1, 1, (self.num_train_users, self.num_factors))
    self.V = np.random.uniform(-1, 1, (self.num_items, self.num_factors))

    rmse = self.evaluate()
    prev_rmse = 1000
    rounds = 0
    num_iters = 1 # change back to 20
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
        print("User iter {}".format(i))
          
      # Optimize V
      for i in range(num_iters):
        for item in self.V_index:
          step = self.get_v_step(item)
          self.V[item] += self.lrate * step
        print("Item iter {}".format(i))
          
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
      print("Regularization: {}".format(self.reg))
      print("Final RMSE: {}\n".format(rmse))
      




class LeastSquares(MatrixFactorization):
  def __init__(self, train, test, num_items, num_factors=30, lrate=0.1, reg=0.1):
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
  def __init__(self, train, test, num_items, num_factors=30, lrate=0.1, reg=0.1):
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




class HuberLoss(MatrixFactorization):
  def __init__(self, train, test, num_items, num_factors=30, lrate=0.1, reg=0.1, delta=1):
    MatrixFactorization.__init__(self, train, test, num_items, num_factors, lrate, reg)
    self.delta = delta
    if verbose:
      print("delta = {}".format(self.delta))

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



def estGeneral1D(X, v, eta):
  v = v / np.linalg.norm(v)
  m = X.shape[0]
  Z = X * v
  np.sort(Z)

  intervalWidth = int(m * ((1 - eta) ** 2))
  lengths = np.zeros(m - intervalWidth + 1)

  for i in range(m - intervalWidth + 1):
    lengths[i] = Z[i + intervalWidth - 1] - Z[i]

  ind = np.argmin(lengths)
  mu = np.mean(Z[ind : ind + intervalWidth])
  return mu


def outRemBall(X, eta):
  m = X.shape[0]
  w = np.ones(m)
  Z = X - np.median(X, axis=0)
  T = np.sum(Z ** 2, axis=1)
  thresh = np.percentile(T, (100 * ((1 - eta) ** 2)))
  w[T > thresh] = 0
  return w


def agnosticMeanGeneral(X, eta):
  n = X.shape[1]
  if n <= 1:
    est = estGeneral1D(X.reshape(-1,), 1, eta)
    return np.array([est])
  
  w = outRemBall(X, eta)
  newX = X[w > 0]

  S = np.cov(newX, rowvar=False)
  # print("S: {}".format(S.shape))
  [D, V] = np.linalg.eigh(S)

  if False not in (np.diff(D) >= 0):
    inds = np.argsort(D)
    V = V[:, inds]

  PW = np.matmul(V[:, :int(n / 2)], V[:, :int(n / 2)].T)
  weightedProjX = np.matmul(newX, PW)
  est1 = np.mean(weightedProjX, axis=0)
  # print("est 1: {}".format(est1.shape))

  QV = V[:, int(n / 2):]
  # print("QV: {}".format(QV.shape))
  # print("newX: {}".format(newX.shape))
  est2 = agnosticMeanGeneral(np.matmul(X, QV), eta)
  # print("est2 shape: {}".format(est2.shape))
  # print("est2: {}".format(est2))
  est2 = np.matmul(est2, QV.T)
  est = est1 + est2
  return est




class HuberGradient(MatrixFactorization):
  def __init__(self, train, test, num_items, num_factors=30, lrate=0.1, reg=0.1, corruption=0.1):
    MatrixFactorization.__init__(self, train, test, num_items, num_factors, lrate, reg)
    self.corruption = corruption
    

  def get_u_step(self, user):
    data = self.train_u[self.U_start[user] : self.U_start[user + 1]]
    vmat = self.V[data[:, 1]]
    preds = np.matmul(vmat, self.U[user])
    return agnosticMeanGeneral(np.multiply((data[:, 2] - preds).reshape((-1, 1)), vmat), self.corruption) - (self.reg * self.U[user])

    
  def get_v_step(self, item):
    data = self.train_v[self.V_start[item] : self.V_start[item + 1]]
    umat = self.U[data[:, 0]]
    preds = np.matmul(umat, self.V[item])
    if len(data) <= 1:
      return np.multiply((data[:, 2] - preds).reshape((-1, 1)), umat).reshape(-1,) - (self.reg * self.V[item])
    return agnosticMeanGeneral(np.multiply((data[:, 2] - preds).reshape((-1, 1)), umat), self.corruption) - (self.reg * self.V[item])




class WeightedMean(MatrixFactorization):
  def __init__(self, train, test, num_items, num_factors=30, lrate=0.1, reg=0.1):
    MatrixFactorization.__init__(self, train, test, num_items, num_factors, lrate, reg)


  def get_u_step(self, user):
    data = self.train_u[self.U_start[user] : self.U_start[user + 1]]
    vmat = self.V[data[:, 1]]
    preds = np.matmul(vmat, self.U[user])
    residuals = data[:, 2] - preds
    res_norm = (residuals - np.mean(residuals)) / np.std(residuals)
    densities = np.exp(-(res_norm ** 2) / 2) / np.sqrt(2 * np.pi)
    weights = densities / np.sum(densities)
    grads = np.multiply((data[:, 2] - preds).reshape((-1, 1)), vmat) - (self.reg * self.U[user])
    return np.average(grads, axis=0, weights=weights)

    
  def get_v_step(self, item):
    data = self.train_v[self.V_start[item] : self.V_start[item + 1]]
    umat = self.U[data[:, 0]]
    preds = np.matmul(umat, self.V[item])
    residuals = data[:, 2] - preds
    grads = np.multiply(residuals.reshape((-1, 1)), umat) - (self.reg * self.V[item])
    res_std = np.std(residuals)
    if res_std > 0:
      res_norm = (residuals - np.mean(residuals)) / res_std
      densities = np.exp(-(res_norm ** 2) / 2) / np.sqrt(2 * np.pi)
      weights = densities / np.sum(densities)
      return np.average(grads, axis=0, weights=weights)
    else:
      return np.mean(grads, axis=0)


        


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
     

  with open("../results/average_attack/trial{}/trial{}_{}.pkl".format(trial_num, trial_num, ModelClass.__name__), "wb+") as f:
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
        ModelClass = LeastSquares
      elif args[1] == "lad":
        print("Training least abs dev model")
        ModelClass = LeastAbsDev
      elif args[1] == "hl":
        print("Training Huber loss model")
        ModelClass = HuberLoss
      elif args[1] == "hg":
        print("Training Huber gradient model")
        ModelClass = HuberGradient
      elif args[1] == "wm":
        print("Training weighted mean model")
        ModelClass = WeightedMean

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
        average_attack(LeastSquares)
      elif args[2] == "lad":
        print("Attacking least abs dev model")
        average_attack(LeastAbsDev)
      elif args[2] == "hl":
        print("Attacking Huber loss model")
        average_attack(HuberLoss)
      elif args[2] == "wm":
        print("Attacking weighted mean model")
        average_attack(WeightedMean)
      elif args[2] == "all":
        print("Attacking all models")
        average_attack(LeastSquares)
        average_attack(LeastAbsDev)
        average_attack(HuberLoss)

  else:
    X = np.random.standard_normal((100, 30))
    X[range(0, 100, 10)] = np.random.uniform(-10, 10, (10, 30))
    est = agnosticMeanGeneral(X, 0.1)
    print("mean norm: {}".format(np.linalg.norm(np.mean(X, axis=0))))
    print("median norm: {}".format(np.linalg.norm(np.median(X, axis=0))))
    print("est norm: {}".format(np.linalg.norm(est)))



    
    