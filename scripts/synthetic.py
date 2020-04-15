import sys
import numpy as np
import pandas as pd
import alt_min



def run_experiment(ModelClass, n, r, p, eta, c, reg=None):
  U = np.random.standard_normal((n, r))
  V = np.random.standard_normal((r, n))
  L = np.matmul(U, V)

  mask_eta = np.random.rand(L.shape[0], L.shape[1])
  S = np.random.uniform(-c, c, L.shape)
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

  if reg == None:
    if ModelClass.__name__ == "HuberGradient":
      model = ModelClass(train, test, n, corruption=eta)
    else:
      model = ModelClass(train, test, n)
  else:
    if ModelClass.__name__ == "HuberGradient":
      model = ModelClass(train, test, n, corruption=eta, reg=reg)
    else:
      model = ModelClass(train, test, n, reg=reg)
    print("Regularization: {}".format(reg))
    
  model.alt_min()



if __name__ == '__main__':
  args = sys.argv
  if len(args) >= 2:
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
    else:
      ModelClass = alt_min.LeastSquares

    n = 200
    r = 20
    p = 0.2
    eta = 0.1
    c = 100

    if len(args) >= 4 and args[2] == "reg":
      run_experiment(ModelClass, n, r, p, eta, c, float(args[3]))
    else:
      run_experiment(ModelClass, n, r, p, eta, c)
  else:
    X = np.random.standard_normal((100, 30))
    X[range(0, 100, 10)] = np.random.uniform(-100, 100, (10, 30))
    est = alt_min.agnosticMeanGeneral(X, 0.1)
    print("mean norm: {}".format(np.linalg.norm(np.mean(X, axis=0))))
    print("median norm: {}".format(np.linalg.norm(np.median(X, axis=0))))
    print("est norm: {}".format(np.linalg.norm(est)))
    

    

    


      

    