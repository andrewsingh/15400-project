import sys
import numpy as np
import random
import pandas as pd
import pickle
import time


# verbose = False


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
      square_error += np.sum((user_data[:, 2] - np.matmul(self.V[user_data[:, 1].astype(int)], self.U[user])) ** 2)
    return np.sqrt(square_error / len(data))


  def evaluate_item(self, item):
    data = self.test_v[self.V_start_test[item] : self.V_start_test[item + 1]]
    return np.sqrt(np.sum((data[:, 2] - np.matmul(self.U[data[:, 0].astype(int)], self.V[item])) ** 2) / len(data))


  def get_u_step(self, user):
    pass


  def get_v_step(self, item):
    pass


  def alt_min(self, verbose):
    np.random.seed(0)
    self.U = np.random.uniform(-1, 1, (self.num_train_users, self.num_factors))
    self.V = np.random.uniform(-1, 1, (self.num_items, self.num_factors))

    rmse = self.evaluate()
    prev_rmse = 1000
    min_rmse = rmse
    rounds = 0
    num_iters = 20
    threshold = -0.0001

    while rmse - prev_rmse < threshold or rounds < 100:
      if verbose:
        t0 = time.time()
      prev_rmse = rmse
      
      # Optmize U
      for i in range(num_iters):
        for user in range(self.num_train_users):
          step = self.get_u_step(user)
          # grad_norm = np.linalg.norm(step)
          # print("grad norm: {}".format(np.linalg.norm(step)))
          # if (grad_norm >= 3):
          #   print("grad norm: {}".format(np.linalg.norm(step)))
          #   print(step)
          self.U[user] += self.lrate * step
          # print("grad norm: {}".format(np.linalg.norm(self.lrate * step)))
        # print("User iter {}".format(i))
        
          
      # Optimize V
      for i in range(num_iters):
        for item in self.V_index:
          step = self.get_v_step(item)
          # print("grad norm: {}".format(np.linalg.norm(step)))
          self.V[item] += self.lrate * step
        # print("Item iter {}".format(i))
          
      rmse = self.evaluate()
      if rmse < min_rmse:
        min_rmse = rmse

      rounds += 1

      if verbose:
        t1 = time.time()
        # train_rmse = self.evaluate(False)
        print("\n==================== ROUND {} ====================\nRMSE: {}\nPrev RMSE: {}\nDiff: {}\nExecution time: {}\n" \
          .format(rounds, round(rmse, 4), round(prev_rmse, 4), round(rmse - prev_rmse, 4), round(t1 - t0, 2)))
      
        print("max U: {}\nmin U: {}\navg U: {}\n".format(np.amax(self.U), np.amin(self.U), np.mean(self.U)))
        print("max V: {}\nmin V: {}\navg V: {}\n".format(np.amax(self.V), np.amin(self.V), np.mean(self.V)))
        print("Factors: {}".format(self.num_factors))
        print("Regularization: {}".format(self.reg))
        print("RMSE: {}\n".format(rmse))

    return min_rmse    





class LeastSquares(MatrixFactorization):
  def __init__(self, train, test, num_items, num_factors, reg, lrate=0.1):
    MatrixFactorization.__init__(self, train, test, num_items, num_factors, lrate, reg)


  def get_u_step(self, user):
    data = self.train_u[self.U_start[user] : self.U_start[user + 1]]
    vmat = self.V[data[:, 1].astype(int)]
    preds = np.matmul(vmat, self.U[user])
    return np.mean(np.multiply((data[:, 2] - preds).reshape((-1, 1)), vmat) - (self.reg * self.U[user]), axis=0)

    
  def get_v_step(self, item):
    data = self.train_v[self.V_start[item] : self.V_start[item + 1]]
    umat = self.U[data[:, 0].astype(int)]
    preds = np.matmul(umat, self.V[item])
    return np.mean(np.multiply((data[:, 2] - preds).reshape((-1, 1)), umat) - (self.reg * self.V[item]), axis=0)



class LeastAbsDev(MatrixFactorization):
  def __init__(self, train, test, num_items, num_factors, reg, lrate=0.1):
    MatrixFactorization.__init__(self, train, test, num_items, num_factors, lrate, reg)


  def get_u_step(self, user):
    data = self.train_u[self.U_start[user] : self.U_start[user + 1]]
    vmat = self.V[data[:, 1].astype(int)]
    preds = np.matmul(vmat, self.U[user])
    return np.mean(np.multiply(np.sign(data[:, 2] - preds).reshape((-1, 1)), vmat) - (self.reg * self.U[user]), axis=0)

    
  def get_v_step(self, item):
    data = self.train_v[self.V_start[item] : self.V_start[item + 1]]
    umat = self.U[data[:, 0].astype(int)]
    preds = np.matmul(umat, self.V[item])
    return np.mean(np.multiply(np.sign((data[:, 2] - preds)).reshape((-1, 1)), umat) - (self.reg * self.V[item]), axis=0)




class HuberLoss(MatrixFactorization):
  def __init__(self, train, test, num_items, num_factors, reg, lrate=0.1, delta=1):
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



def estGeneral1D(X, eta):
  m = X.shape[0]
  Z = np.sort(X)
  
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
    est = estGeneral1D(X.reshape(-1,), eta)
    return np.array([est])
  
  w = outRemBall(X, eta)
  newX = X[w > 0]

  S = np.cov(newX, rowvar=False)
  [_, V] = np.linalg.eigh(S)

  PW = np.matmul(V[:, :int(n / 2)], V[:, :int(n / 2)].T)
  weightedProjX = np.matmul(newX, PW)
  est1 = np.mean(weightedProjX, axis=0)

  QV = V[:, int(n / 2):]
  est2 = agnosticMeanGeneral(np.matmul(X, QV), eta)
  est2 = np.matmul(est2, QV.T)
  est = est1 + est2
  # print("ratio: {}".format(np.linalg.norm(est / np.mean(X, axis=0))))
  return est




class HuberGradient(MatrixFactorization):
  def __init__(self, train, test, num_items, num_factors, reg, corruption, lrate=0.0183):
    MatrixFactorization.__init__(self, train, test, num_items, num_factors, lrate, reg)
    self.corruption = corruption
    

  def get_u_step(self, user):
    data = self.train_u[self.U_start[user] : self.U_start[user + 1]]
    vmat = self.V[data[:, 1].astype(int)]
    preds = np.matmul(vmat, self.U[user])
    return agnosticMeanGeneral(np.multiply((data[:, 2] - preds).reshape((-1, 1)), vmat) - (self.reg * self.U[user]), self.corruption) 

    
  def get_v_step(self, item):
    data = self.train_v[self.V_start[item] : self.V_start[item + 1]]
    umat = self.U[data[:, 0].astype(int)]
    preds = np.matmul(umat, self.V[item])
    # if len(data) <= 1:
    #   return np.multiply((data[:, 2] - preds).reshape((-1, 1)), umat).reshape(-1,) - (self.reg * self.V[item])
    return agnosticMeanGeneral(np.multiply((data[:, 2] - preds).reshape((-1, 1)), umat) - (self.reg * self.V[item]), self.corruption) 






        










    
    