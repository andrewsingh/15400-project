import sys
import numpy as np
import random
import pandas as pd
import pickle
import time


num_users = 6040
num_items = 3706

train = pd.read_pickle("../data/ml-1m-split/train.pkl")
test = pd.read_pickle("../data/ml-1m-split/test.pkl")
train_u = train.sort_values(by="user", axis=0).reset_index(drop=True).drop(["item_id", "timestamp"], axis=1).to_numpy()
train_v = train.sort_values(by="item", axis=0).reset_index(drop=True).drop(["item_id", "timestamp"], axis=1).to_numpy()
test_u = test.sort_values(by="user", axis=0).reset_index(drop=True).drop(["item_id", "timestamp"], axis=1).to_numpy()
U_freqs = train.groupby("user").size().values
U_freqs_test = test.groupby("user").size().values
V_group = train.groupby("item").size()
V_index = V_group.index.values
V_freqs = np.zeros(num_items, dtype="int")
for i in V_group.index:
  V_freqs[i] = V_group[i]
  
U_start = np.cumsum(U_freqs)
U_start_test = np.cumsum(U_freqs_test)
V_start = np.cumsum(V_freqs)
U_start = np.insert(U_start, 0, 0)
U_start_test = np.insert(U_start_test, 0, 0)
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
        start = U_start_test
      else:
        data = train_u
        start = U_start

      square_error = 0
      for user in range(num_users):
        user_data = data[start[user] : start[user + 1]]
        square_error += np.sum((user_data[:, 2] - np.matmul(V[user_data[:, 1]], U[user])) ** 2)

      return np.sqrt(square_error / len(data))



    def get_u_step(user):
      data = train_u[U_start[user] : U_start[user + 1]]
      vmat = V[data[:, 1]]
      preds = np.matmul(vmat, U[user])
      return np.mean(np.multiply((data[:, 2] - preds).reshape((-1, 1)), vmat) - (reg * U[user]), axis=0)

    
    def get_v_step(item):
      data = train_v[V_start[item] : V_start[item + 1]]
      umat = U[data[:, 0]]
      preds = np.matmul(umat, V[item])
      return np.mean(np.multiply((data[:, 2] - preds).reshape((-1, 1)), umat) - (reg * V[item]), axis=0)


    train_rmse = evaluate()
    rmse = evaluate(True)
    prev_rmse = 100
    outer = 0
    reg = 0.1

    print("Initial train RMSE: {}\nInitial test RMSE: {}\n".format(train_rmse, rmse))


    while rmse - prev_rmse < 0:
        t0 = time.time()
        prev_rmse = rmse
        
        # Optmize U
        inner = 0
        step_size = 100
        # while abs(step_size) > step_limit:
        for i in range(20):
            step_size = 0
            for user in range(num_users):
              step = get_u_step(user)
              U[user] += lrate * step
              step_size += np.sum(lrate * np.abs(step))
            
            step_size /= (num_users * num_factors)
            inner += 1
            
        
        # Optimize V
        inner = 0
        step_size = 100
        for i in range(20):
        # while abs(step_size) > step_limit:            
            step_size = 0
            for item in V_index:
              step = get_v_step(item)
              V[item] += lrate * step
              step_size += np.sum(lrate * np.abs(step))
            
            step_size /= (num_items * num_factors)
            inner += 1

        train_rmse = evaluate()
        rmse = evaluate(True)

        t1 = time.time()

        print("\n======================== ROUND {} ========================\nRMSE: {}\nPrev RMSE: {}\nDiff: {}\nTrain RMSE: {}".format(outer, rmse, prev_rmse, rmse - prev_rmse, train_rmse))
        print("Execution time: {}\n".format(t1 - t0))
        outer += 1
    


 
if __name__ == '__main__':
    args = sys.argv
    if len(args) >= 3:
        alt_min(int(args[1]), float(args[2]))
        
    
    