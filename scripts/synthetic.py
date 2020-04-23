import sys
import numpy as np
import pandas as pd
import alt_min
import argparse



def generate_prob_mask(n, p):
  row_visible = np.zeros(n).astype(int)
  col_visible = np.zeros(n).astype(int)
  prob_mask = np.zeros((n, n))
  visible = 0
  for entry in np.random.permutation(n * n):
    row = (int)(entry / n)
    col = entry % n
    if (row_visible[row] < n * p) and (col_visible[col] < n * p):
      prob_mask[row][col] = 1
      visible += 1
      row_visible[row] += 1
      col_visible[col] += 1
    if visible == n * n * p:
      break
  return prob_mask


def generate_noise_mask(n, p, eta, prob_mask):
  row_corruptions = np.zeros(n).astype(int)
  col_corruptions = np.zeros(n).astype(int)
  noise_mask = np.zeros((n, n))
  visible_entries = []
  for i in range(n):
    for j in range(n):
      if prob_mask[i][j] != 0:
        visible_entries.append((i, j))
  visible_entries = np.random.permutation(visible_entries)
  corruptions = 0
  for (row, col) in visible_entries:
    if (row_corruptions[row] < n * p * eta) and (col_corruptions[col] < n * p * eta):
      noise_mask[row][col] = 1
      corruptions += 1
      row_corruptions[row] += 1
      col_corruptions[col] += 1
    if corruptions == n * n * p * eta:
      break
  return noise_mask


def run_experiment(ModelClass, reg, n, r, p, eta, c, b, verbose):
  U = np.random.standard_normal((n, r))
  V = np.random.standard_normal((r, n))
  L = np.matmul(U, V)

  prob_mask = generate_prob_mask(n, p)
  noise_mask = generate_noise_mask(n, p, eta, prob_mask)
  S_obs = np.random.uniform(-c + b, c + b, L.shape) * noise_mask
  M_obs = (L * prob_mask) + S_obs
  print("Observations: {}".format(len(M_obs[M_obs != 0])))
  print("Corruptions: {}".format(len(S_obs[S_obs != 0])))

  user_col = []
  item_col = []
  rating_col = []
  for i in range(M_obs.shape[0]):
    for j in range(M_obs.shape[1]):
      rating = M_obs[i][j]
      if rating != 0:
        user_col.append(i)
        item_col.append(j)
        rating_col.append(rating)
  train = pd.DataFrame.from_dict({"user": user_col, "item": item_col, "rating": rating_col})

  user_col = []
  item_col = []
  rating_col = []
  for i in range(L.shape[0]):
    for j in range(L.shape[1]):
      rating = L[i][j]
      user_col.append(i)
      item_col.append(j)
      rating_col.append(rating)
  test = pd.DataFrame.from_dict({"user": user_col, "item": item_col, "rating": rating_col})

  if ModelClass.__name__ in ["HuberGradient", "RemoveOutliers"]:
    model = ModelClass(train, test, n, num_factors=r, reg=reg, corruption=eta)
  else:
    model = ModelClass(train, test, n, num_factors=r, reg=reg)
    
  return model.alt_min(verbose)



if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Alt-Min matrix completion algorithm')

  parser.add_argument("-n", dest="n", type=int)
  parser.add_argument("-r", dest="r", type=int)
  parser.add_argument("-p", dest="p", type=float)
  parser.add_argument("-eta", dest="eta", type=float)
  parser.add_argument("-c", dest="c", type=float, default=50)
  parser.add_argument("-b", dest="b", type=float, default=0)
  parser.add_argument("-m", dest="mclass")
  parser.add_argument("-reg", dest="reg", type=float)
  parser.add_argument("-v", action="store_true")
  
  args = parser.parse_args() 

  if args.mclass == "ls":
    print("Training least squares model")
    ModelClass = alt_min.LeastSquares
  elif args.mclass == "lad":
    print("Training least abs dev model")
    ModelClass = alt_min.LeastAbsDev
  elif args.mclass == "hl":
    print("Training Huber loss model")
    ModelClass = alt_min.HuberLoss
  elif args.mclass == "hg":
    print("Training Huber gradient model")
    ModelClass = alt_min.HuberGradient
  elif args.mclass == "ro":
    print("Training remove outliers model")
    ModelClass = alt_min.RemoveOutliers
  elif args.mclass == "rone":
    print("Training remove outliers no eta model")
    ModelClass = alt_min.RemoveOutliersNoEta
  else:
    ModelClass = alt_min.LeastSquares

  rmse = run_experiment(ModelClass, args.reg, args.n, args.r, args.p, args.eta, args.c, args.b, args.v)
  print("Experiment complete\nargs: {}".format(args))
  with open("../results/synthetic/exp_log.txt", "a+") as f:
    f.write("{} final RMSE: {}\n".format(args, rmse))








    

    


      

    