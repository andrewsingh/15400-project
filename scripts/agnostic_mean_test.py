import sys
import numpy as np
import pandas as pd
import alt_min
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Alt-Min matrix completion algorithm')

    parser.add_argument("-n", dest="n", type=int)
    parser.add_argument("-d", dest="d", type=int)
    parser.add_argument("-eta", dest="eta", type=float)
    parser.add_argument("-c", dest="c", type=float, default=50)
    parser.add_argument("-b", dest="b", type=float, default=0)
    parser.add_argument("-t", dest="t", type=int, default=1)
    parser.add_argument("-v", action="store_true")

    args = parser.parse_args()

    est_norms = np.zeros(args.t)
    med_norms = np.zeros(args.t)
    mean_norms = np.zeros(args.t)

    for i in range(args.t):            
        X = np.random.standard_normal((args.n, args.d))
        corrupt_idx = np.random.choice(args.n, (int)(args.eta * args.n), replace=False)
        X[corrupt_idx] = np.random.uniform(-args.c + args.b, args.c + args.b, (len(corrupt_idx), args.d))
        est = alt_min.agnosticMeanGeneral(X, args.eta)
        mean_norm = np.linalg.norm(np.mean(X, axis=0))
        med_norm = np.linalg.norm(np.median(X, axis=0))
        est_norm = np.linalg.norm(est)
        mean_norms[i] = mean_norm
        med_norms[i] = med_norm
        est_norms[i] = est_norm
        if args.v:
            print("\nTrial {}\nest norm: {}\nmedian norm: {}\nmean norm: {}".format(i + 1, round(est_norm, 4), round(med_norm, 4), round(mean_norm, 4)))


    print("Avg est norm: {}".format(round(np.mean(est_norms), 4)))
    print("Avg med norm: {} ({}x est, stdev {}x)".format(round(np.mean(med_norms), 4), round(np.mean(med_norms / est_norms), 2), round(np.std(med_norms / est_norms), 2)))
    print("Avg mean norm: {} ({}x est, stdev {}x)".format(round(np.mean(mean_norms), 4), round(np.mean(mean_norms / est_norms), 2), round(np.std(mean_norms / est_norms), 2)))