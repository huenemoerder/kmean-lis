# Towards a Learned Index Structure for Approximate Nearest Neighbor Search Query Processing

This is the repo for our paper "Towards a Learned Index Structure for Approximate Nearest Neighbor Search Query Processing". In this short paper, we outline the idea of applying the concept of a learned index structure to approximate nearest neighbor query processing. We discuss different data partitioning approaches and show how the task of identifying the disc pages of potential hits for a given query can be solved by a predictive machine learning model. In a preliminary experimental case study we evaluate and discuss the general applicability of different partitioning approaches as well as of different predictive models.

You can find our experiments with synthetic data in the following notebooks:
1. [Clustered data with using kmeans as paritioning](Experiment_01__kmeans_clustered.ipynb)
2. [A single gaussian blob using kmeans as paritioning](Experiment_01__kmeans_uniform.ipynb)
3. [Clustered data with using the leaves of a kd-tree as paritioning](Experiment_03__KD_Tree_clustered.ipynb)
4. [A single gaussian blob using the leaves of a kd-tree as paritioning](Experiment_04__KD_Tree_Uniform.ipynb)

Additionally we tried our method on MNIST using a simple Autoencoder for feature reduction.
You can find the notebooks (here)[MNIST-LIS.ipynb] and (here)[MNIST-LIS-KD_Tree.ipynb]
The figures illustrating our results are generated (here)[Analysis.ipynb
