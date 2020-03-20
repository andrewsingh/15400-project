import sys
import numpy as np
import random
import pandas as pd
import pickle


train_df = pd.read_pickle("../data/ml-1m-split/train.pkl")
# train_shuffled = train_df.sample(frac=1).reset_index(drop=True)
# test = pd.read_pickle("../data/ml-1m-split/test.pkl")

num_users = 6040
num_items = 3706

train_u = np.zeros((len(train_df), 3))
train_m = np.zeros((len(train_df), 3))




for (idx, [user, item, _, rating, _]) in train_df.sort_values(by="user", axis=0).reset_index(drop=True):
    train_u[idx] = np.array([user, item, rating])


# avg_item_ratings = np.zeros(num_items)
# for (item, df) in train.groupby("item"):
#     avg_item_ratings[item] = np.mean(df["rating"].values)


def alt_min(num_factors, lrate):
    U = np.random.uniform(0, 0.6, (num_users, num_factors))
    M = np.random.uniform(0, 0.6, (num_items, num_factors))
    U_freqs = train_df.groupby("user").size().values.reshape(-1, 1)
    M_freqs = train_df.groupby("item").size().values.reshape(-1, 1)
    rmse = 100
    prev_rmse = 0
    round = 0

    while abs(rmse - prev_rmse) > 0.001:
        prev_rmse = rmse

        # Optmize U
        step_size = 100
        iter = 0
        u_rmse = 100
        u_prev_rmse = 0
        while abs(u_rmse - u_prev_rmse) > 0.001:
            u_prev_rmse = u_rmse
            U_step = np.zeros((num_users, num_factors))
            total_loss = 0
            for (idx, [user, item, _, rating, _]) in train_df.iterrows():
                residual = rating - np.dot(U[user], M[item])
                total_loss += residual ** 2
                U_step[user] += residual * M[item]
            
            U_step = lrate * (U_step / U_freqs)
            U += U_step
            step_size = np.linalg.norm(U_step)
            u_rmse = np.sqrt(total_loss / len(train_shuffled))
            print("USER ITER {}\nStep size: {}\nRMSE: {}\n".format(iter, step_size, u_rmse))
            iter += 1
            if iter > 1:
                break

        if iter > 1:
            break
        
        # Optimize M
        step_size = 100
        iter = 0
        m_prev_rmse = 0
        while abs(rmse - m_prev_rmse) > 0.001:
            m_prev_rmse = rmse
            M_step = np.zeros((num_users, num_factors))
            total_loss = 0
            for (idx, [user, item, _, rating, _]) in train_shuffled.iterrows():
                residual = rating - np.dot(U[user], M[item])
                total_loss += residual ** 2
                M_step[item] += residual * U[user]
            
            M_step = lrate * (M_step / M_freqs)
            M += M_step
            step_size = np.linalg.norm(M_step)
            rmse = np.sqrt(total_loss / len(train_shuffled))
            print("ITEM ITER {}\nStep size: {}\nRMSE: {}\n".format(iter, step_size, rmse))
            iter += 1


        print("\n============ ROUND {} ============\nRMSE: {}\nPrev RMSE: {}\nDiff: {}\n".format(round, rmse, prev_rmse, rmse - prev_rmse))
        round += 1
    


 
if __name__ == '__main__':
    args = sys.argv
    if len(args) >= 3:
        alt_min(int(args[1]), float(args[2]))
        
    
    