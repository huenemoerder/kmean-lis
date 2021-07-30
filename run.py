from utils import BaselineClassifier, evaluate, nn_accuracy, run
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import os 
import warnings

warnings.filterwarnings('ignore')

ns = [5000, 10000, 30000, 50000, 100000]
for n in ns:
    kmeans_results = []
    kd_results = []
    for seed in tqdm(range(5)):
        kmeans_result, _ = run(n_samples=n, 
                        n_features=20, 
                        n_clusters=1,
                        cluster_std=1.0,
                        k=int(n/1000),
                        n_test=100, 
                        random_seed=seed, 
                        kd_tree=False)
        kmeans_results.append(kmeans_result)

        kd_result, _ = run(n_samples=n, 
                        n_features=20, 
                        n_clusters=1,
                        cluster_std=1.0,
                        k=int(n/1000),
                        n_test=100, 
                        random_seed=seed, 
                        kd_tree=True)
        kd_results.append(kd_result)

    hdr = False  if os.path.isfile('results/kmeans_uniform_rel_error.csv') else True
    pd.DataFrame(kmeans_results).to_csv(f'results/kmeans_uniform_rel_error.csv', mode='a', header=hdr)

    hdr = False  if os.path.isfile('results/kdtree_uniform_rel_error.csv') else True
    pd.DataFrame(kd_results).to_csv(f'results/kdtree_uniform_rel_error.csv', mode='a', header=hdr)
