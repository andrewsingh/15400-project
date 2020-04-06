import sys
import numpy as np
import random
import pandas as pd
import pickle


num_users = 6040
num_items = 3706

train = pd.read_pickle("../data/ml-1m-split/train.pkl")
test = pd.read_pickle("../data/ml-1m-split/test.pkl")
train_u = train.sort_values(by="user", axis=0).reset_index(drop=True).drop(["item_id", "timestamp"], axis=1).to_numpy()
train_v = train.sort_values(by="item", axis=0).reset_index(drop=True).drop(["item_id", "timestamp"], axis=1).to_numpy()
test_u = test.sort_values(by="user", axis=0).reset_index(drop=True).drop(["item_id", "timestamp"], axis=1).to_numpy()
U_freqs = train.groupby("user").size().values
V_group = train.groupby("item").size()
V_index = V_group.index.values
V_freqs = np.zeros(num_items, dtype="int")
for i in V_group.index:
  V_freqs[i] = V_group[i]
  
U_start = np.cumsum(U_freqs)
V_start = np.cumsum(V_freqs)
U_start = np.insert(U_start, 0, 0)
V_start = np.insert(V_start, 0, 0)


def alt_min(num_factors, lrate):
    # U = np.linalg.svd()
    U = np.random.uniform(-1, 1, (num_users, num_factors))
    V = np.random.uniform(-1, 1, (num_items, num_factors))

    print("max U: {}\nmin U: {}\navg U: {}\n".format(np.amax(U), np.amin(U), np.mean(U)))
    print("max V: {}\nmin V: {}\navg V: {}".format(np.amax(V), np.amin(V), np.mean(V)))

    # U = np.random.standard_normal((num_users, num_factors))
    # U /= np.linalg.norm(U, axis=1).reshape((-1 ,1))
    # V = np.random.standard_normal((num_items, num_factors))
    # V /= np.linalg.norm(V, axis=1).reshape((-1 ,1))
    
    def evaluate(test=False):
      if test:
        data = test_u
      else:
        data = train_u
      return np.sqrt(np.mean(np.apply_along_axis(lambda x: (x[2] - np.dot(U[x[0]], V[x[1]])) ** 2, 1, data), axis=0))
    
    rmse = evaluate(True)
    prev_rmse = 100
    outer = 0
    reg = 0.1

    print("Initial train RMSE: {}\nInitial test RMSE: {}\n".format(evaluate(), rmse))
    

    def get_u_step_fast(user):
      data = train_u[U_start[user] : U_start[user + 1]]
      umat = np.repeat(U[user].reshape((1, -1)), len(data), axis=0)
      vmat = V[data[:, 1]]
      rvec = data[:, 2].reshape((-1, 1))
      preds = np.diag(np.matmul(umat, vmat.T)).reshape((-1, 1))
      return np.mean(np.multiply((rvec - preds), vmat) - (reg * umat), axis=0)
    
    
    def get_v_step_fast(item):
      data = train_v[V_start[item] : V_start[item + 1]]
      vmat = np.repeat(V[item].reshape((1, -1)), len(data), axis=0)
      umat = U[data[:, 0]]
      rvec = data[:, 2].reshape((-1, 1))
      preds = np.diag(np.matmul(umat, vmat.T)).reshape((-1, 1))
      return np.mean(np.multiply((rvec - preds), umat) - (reg * vmat), axis=0)
    
         
    while rmse - prev_rmse < 0:
        prev_rmse = rmse
        
        # Optmize U
        inner = 0
        step_size = 100
        # while abs(step_size) > step_limit:
        for i in range(10):
            step_size = 0
            for user in range(num_users):
              step = get_u_step_fast(user)
              U[user] += lrate * step
              step_size += np.sum(lrate * np.abs(step))
            
            step_size /= (num_users * num_factors)
            
            print("\nUSER ITER {}\navg step size: {}".format(inner, step_size))
            print("max U: {}\nmin U: {}\navg U: {}".format(np.amax(U), np.amin(U), np.mean(U)))
            inner += 1
            
        
        # Optimize V
        inner = 0
        step_size = 100
        for i in range(10):
        # while abs(step_size) > step_limit:            
            step_size = 0
            for item in V_index:
              step = get_v_step_fast(item)
              V[item] += lrate * step
              step_size += np.sum(lrate * np.abs(step))
            
            step_size /= (num_items * num_factors)
            
            print("\nITEM ITER {}\navg step size: {}".format(inner, step_size))
            print("max V: {}\nmin V: {}\navg V: {}".format(np.amax(V), np.amin(V), np.mean(V)))
            inner += 1

        rmse = evaluate(True)

        print("\n============ ROUND {} ============\nRMSE: {}\nPrev RMSE: {}\nDiff: {}\nTrain RMSE: {}\n".format(outer, rmse, prev_rmse, rmse - prev_rmse, evaluate()))
        outer += 1
    


 
if __name__ == '__main__':
    args = sys.argv
    if len(args) >= 3:
        alt_min(int(args[1]), float(args[2]))
        
    
    