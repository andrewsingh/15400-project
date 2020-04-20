import sys
import numpy as np
import pandas as pd
import alt_min
import argparse



def run_experiment(ModelClass, reg, n, r, p, eta, c, b, verbose):
  U = np.random.standard_normal((n, r))
  V = np.random.standard_normal((r, n))
  L = np.matmul(U, V)

  mask_eta = np.random.rand(L.shape[0], L.shape[1])
  S = np.random.uniform(-c + b, c + b, L.shape)
  S[mask_eta < 1 - eta] = 0

  M = L + S
  mask_p = np.random.rand(M.shape[0], M.shape[1])
  mask_p[mask_p < 1 - p] = 0
  mask_p[mask_p != 0] = 1
  M_obs = np.multiply(M, mask_p)

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
  else:
    ModelClass = alt_min.LeastSquares

  rmse = run_experiment(ModelClass, args.reg, args.n, args.r, args.p, args.eta, args.c, args.b, args.v)
  print("Experiment complete\nargs: {}".format(args))
  with open("../results/synthetic/exp_log.txt", "a+") as f:
    f.write("{} final RMSE: {}\n".format(args, rmse))








    

    


      

    