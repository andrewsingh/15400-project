import sys
import numpy as np
import random
import pandas as pd
import pickle


num_users = 6040
num_items = 3706

train = pd.read_pickle("../data/ml-1m-split/train.pkl")
train_u = train.sort_values(by="user", axis=0).reset_index(drop=True).drop(["item_id", "timestamp"], axis=1).to_numpy()
train_v = train.sort_values(by="item", axis=0).reset_index(drop=True).drop(["item_id", "timestamp"], axis=1).to_numpy()
U_freqs = train.groupby("user").size().values
V_group = train.groupby("item").size()
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
    
    def evaluate():
      return np.sqrt(np.mean(np.apply_along_axis(lambda x: (x[2] - np.dot(U[x[0]], V[x[1]])) ** 2, 1, train_u), axis=0))
    
    rmse = evaluate()
    prev_rmse = 0
    outer = 0
    
    print("Initial RMSE: {}\n".format(rmse))
    
    def get_u_step(user):
      # print("{}: {}".format(user, len(train_u[U_start[user] : U_start[user + 1]])))
      return np.mean(np.apply_along_axis(lambda x: (x[2] - np.dot(U[user], V[x[1]])) * V[x[1]], 1, train_u[U_start[user] : U_start[user + 1]]), axis=0)

    def get_v_step(item):
      # print("{}: {}".format(item, len(train_v[V_start[item] : V_start[item + 1]])))
      return np.mean(np.apply_along_axis(lambda x: (x[2] - np.dot(U[x[0]], V[item])) * U[x[0]], 1, train_v[V_start[item] : V_start[item + 1]]), axis=0)
         
    step_limit = 0.03

    while abs(rmse - prev_rmse) > 0.001:
        prev_rmse = rmse
        
        # Optmize U
        inner = 0
        step_size = 100
        # while abs(step_size) > step_limit:
        for i in range(10):
            step_size = 0
            for user in range(num_users):
              step = get_u_step(user)
              U[user] += lrate * step
              step_size += np.sum(lrate * np.abs(step))
            
            step_size /= (num_users * num_factors)
            
            print("USER ITER {}\navg step size: {}\n".format(inner, step_size))
            print("max U: {}\nmin U: {}\navg U: {}".format(np.amax(U), np.amin(U), np.mean(U)))
            inner += 1
            
        
        # Optimize V
        inner = 0
        step_size = 100
        for i in range(10):
        # while abs(step_size) > step_limit:            
            step_size = 0
            for item in V_group.index:
              step = get_v_step(item)
              V[item] += lrate * step
              step_size += np.sum(lrate * np.abs(step))
            
            step_size /= (num_items * num_factors)
            
            print("ITEM ITER {}\navg step size: {}\n".format(inner, step_size))
            print("max V: {}\nmin V: {}\navg V: {}".format(np.amax(V), np.amin(V), np.mean(V)))
            inner += 1

        rmse = evaluate()

        print("\n============ ROUND {} ============\nRMSE: {}\nPrev RMSE: {}\nDiff: {}\n".format(outer, rmse, prev_rmse, rmse - prev_rmse))
        outer += 1
    


 
if __name__ == '__main__':
    args = sys.argv
    if len(args) >= 3:
        alt_min(int(args[1]), float(args[2]))
        
    
    