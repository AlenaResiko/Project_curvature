# curvature
Scalar curvature estimation using the method in paper: "An Intrinsic Approach to Scalar-Curvature Estimation for Point Clouds" (https://arxiv.org/abs/2308.02615)
A. Hickok and A. J. Blumberg

See tutorial.ipynb for some examples. The data files for the tutorial are stored in the example_data folder. Three of the files ("Rdist_10000.npy", "nbr_matrix_10000.npy", and "T_10000.npy") need to be downloaded from:

https://drive.google.com/drive/folders/188mm2Qcm6ewHWXCChpk6ZSL1BRGjxZTy?usp=sharing

and then moved to the example_data folder.


RGG sampling code and ORC estimation
Alena Chan

Those notebooks allow to:
1. Sample points on surface of manifolds: 2d sphere, torus, 3d sphere, flat torus, hyperboloid, poincare disk.
2. Calculate distances between points and build RGG.
3. Visualize 2d manifolds.
4. Compute ORC and OSRC using OlliverRicciCurvature library (although results for flat torus and poincare disk aren't accurate).
5. Computate curvature on edges using optimal transport. Using edge (scalar) curavture, approxima discrete (node) curvature.
