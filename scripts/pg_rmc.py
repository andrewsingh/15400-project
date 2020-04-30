import numpy as np


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



def ncrmc(M, true_r, row, col, p):
    TOL = 1e-1
    incoh = 1
    EPS_S = 1e-3
    EPS = 1e-3
    run_time = 100
    MAX_ITER = 70

    (m1, m2) = M.shape
    D_t = M[row][col]

    n = np.sqrt(m1 * m2)
    frob_err = [1e9]

    t = 0
    thresh_const = incoh * true_r
    thresh_red = 0.9
    r_hat = 1

    B = D_t
    U_t = np.zeros((m1, 1))
    Sig_t = np.zeros((1, 1))
    V_t = np.zeros((m2, 1))
    (U_t, Sig_t, V_t) = np.linalg.svd(U_t @ Sig_t @ V_t.T + (1 / p) * D_t)
    Sig_t = np.diag(Sig_t)
    SV_t = Sig_t @ V_t.T

    thresh = thresh_const * Sig_t / n

    S_t = []

    while frob_err[t] >= EPS and t < MAX_ITER:
        print("iteration {}\nfrob_err: {}\n".format(t, frob_err[t]))
        t = t + 1

        spL_t = U_t @ SV_t
        D_t = B - spL_t
        idx_s = np.absolute(D_t) >= thresh
        S_t = D_t
        S_t[~idx_s] = 0

        (U_t, Sig_t, V_t) = np.linalg.svd(U_t @ Sig_t @ V_t.T + (1 / p) * (D_t - S_t))
        Sig_t = np.diag(Sig_t)
        sigma_t = Sig_t[r_hat + 1][r_hat + 1]

        U_t = U_t[:, :r_hat]
        Sig_t = Sig_t[:r_hat, :r_hat]
        V_t = V_t[:, :r_hat]
        SV_t = Sig_t @ V_t.T

        thresh = (thresh_const / n) * sigma_t

        spL_t = U_t @ SV_t
        frob_err.append(np.linalg.norm(B - (spL_t + S_t), "fro"))

        if ((frob_err[t - 1] - frob_err[t]) / frob_err[t - 1] <= TOL) and r_hat < true_r:
            r_hat = r_hat + 1
        elif ((frob_err[t - 1] - frob_err[t]) / frob_err[t - 1] <= TOL) and r_hat == true_r:
            thresh_const = thresh_const * thresh_red

    return (U_t, SV_t)



if __name__ == '__main__':
    n = 100
    r = 10
    p = 0.5
    eta = 0
    c = 50
    b = 50

    U = np.random.standard_normal((n, r))
    V = np.random.standard_normal((r, n))
    L = U @ V

    prob_mask = generate_prob_mask(n, p)
    noise_mask = generate_noise_mask(n, p, eta, prob_mask)
    S_obs = np.random.uniform(-c + b, c + b, L.shape) * noise_mask
    M_obs = (L * prob_mask) + S_obs

    # Tried normalizing the matrix first, as this was shown in one
    # of the wrapper functions in the matlab code, but didn't help
    # avg = np.mean(np.mean(M_obs, axis=0))
    # M2 = M_obs - avg
    # (U_t, SV_t) = ncrmc(M2, r, 1, 1, p)
    # L_t = U_t @ SV_t + avg
    (U_t, SV_t) = ncrmc(M_obs, r, 1, 1, p)
    L_t = U_t @ SV_t
    final_frob_err = np.linalg.norm(L_t - L, "fro")
    final_rmse = np.sqrt((final_frob_err ** 2) / (n ** 2))
    print("final frob_err: {}\nfinal RMSE: {}".format(final_frob_err, final_rmse))
