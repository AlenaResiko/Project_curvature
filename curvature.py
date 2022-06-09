# -*- coding: utf-8 -*-
import math
from scipy.special import gamma
import numpy as np
from sklearn.manifold import Isomap
import multiprocessing as mp

class KDE:
    def __init__(self, n, X = None, D = None, kernel = None):
        '''
        n: dimension of manifold
        X: N x d matrix containing N observations in d-dimensional ambient space
        D: distance matrix. Must input either X or D
        kernel: optional. kernel function (e.g. gauss or biweight. Default is biweight)
        '''
        assert (X is not None) or (D is not None)
        self.n = n
        self.X = X
        self.D = D
        if kernel is None:
            self.kernel = KDE.biweight
        else:
            self.kernel = kernel
        if X is not None:
            self.N = X.shape[0]
        else:
            self.N = D.shape[0]
        self.h = KDE.bandwidth(self.N, n)   # Scotts's rule
            
    def __call__(self, i):
        if self.D is not None:
            return sum([self.kernel(self.D[i, j]/self.h, self.n)/(self.N*math.pow(self.h, self.n)) for j in range(self.N)])
        else:
            return sum([self.kernel(np.linalg.norm(y - self.X[i, :])/self.h, self.n)/(self.N*math.pow(self.h, self.n)) for y in self.X])
    
    def gauss(x, n):
        '''
        Returns Gaussian kernel evaluated at point x
        '''
        return (1/math.pow(math.sqrt(2*math.pi), n))*math.exp(-x*x/2)
    
    def density(self):
        with mp.Pool(mp.cpu_count()) as p:
            density = p.map(self, np.arange(self.N))
        return density
    
    def biweight(x, n):
        if -1 < x < 1:
            s = 2*((math.pi)**(n/2))/gamma(n/2)
            normalization = s*(1/n - 2/(n+2) + 1/(n+4))
            return ((1-x**2)**2)/normalization
        else:
            return 0
    
    def epanechnikov(x, n):
        if -1 < x < 1:
            s = 2*((math.pi)**(n/2))/gamma(n/2)
            normalization = s*(1/n - 1/(n+2))
            return (1-x**2)/normalization
        else:
            return 0
        
    def triweight(x, n):
        if -1 < x < 1:
            s = 2*((math.pi)**(n/2))/gamma(n/2)
            normalization = s*(-1/(n+6) + 3/(n+4) - 3/(n+2) + 1/n)
            return ((1-x**2)**3)/normalization
        else:
            return 0
    
    def bandwidth(N, d):
        return N**(-1/(d+4))
    
class scalar_curvature_est:
    def __init__(self, X, n, n_nbrs = 20, kernel = None, density = None, Rdist = None, verbose = True):
        '''
        X: N x d matrix containing N observations in d-dimensional ambient space
        n: Integer. dimension of the manifold
        n_nbrs: Integer. number of neighbors to use for Isomap Riemannian distance estimation (Isomap default is 5 but this is generally way too low)
        density: (optional) density[i] is an estimate of the density at X[i, :]
        Rdist: (optional) N x N matrix of Riemannian distances (exact or precomputed approximate distances).
        '''
        self.X = X
        self.n = n
        self.n_nbrs = n_nbrs
        self.density = density
        self.N = X.shape[0] # number of observations
        self.d = X.shape[1] # ambient dimension
        self.Vn = (math.pi**(self.n/2))/gamma(self.n/2 + 1) # volume of Euclidean unit n-ball
        if Rdist is None:
            self.Rdist = scalar_curvature_est.compute_Rdist(X, n_nbrs)
            if verbose: print("computed Rdist")
        else:
            self.Rdist = Rdist

        self.nearest_nbr_dist = [np.min([self.Rdist[i, j] for j in range(self.N) if j!= i]) for i in range(self.N)]
        if verbose: print("computed nearest neighbor distances")
        
        self.kernel = kernel
        if density is None:
            self.density = scalar_curvature_est.compute_density(n, X, kernel)
            if verbose: print("computed density")
        else:
            self.density = density
        
    def ball_ratios(self, i, rmax = None, rs = None):
        '''
        i: index of observation in X (a row)
        rmax: (optional) positive real number. scale at which we're computing scalar curvature (max radius in the rs sequence, if rs isn't given)
        rs: (optional) increasing sequence of radii
        Must input either rmax or rs.
        
        Returns:
        rs: sequence of radii in range [0, rmax]. 
            If not initially given: r_j is estimated Riemannian distance from x_i 
                to its jth nearest neighbor.
            If given as input: returns given rs
        ball_ratios: sequence of estimaed ratios of geodesic ball volume to 
            euclidean ball volume, for each radius r in rs.
        '''
        assert (rmax is not None) or (rs is not None)
        rs, ball_vols = self.ball_volumes(i, rmax, rs)
        ball_ratios = np.array([ball_vols[j]/(self.Vn*(r**self.n)) for j, r in enumerate(rs)])
        return rs, ball_ratios
    
    def ball_volumes(self, i, rmax = None, rs = None):
        '''
        i: integer. index of observation in X (a row)
        rmax: (optional) positive real number. scale at which we're computing scalar curvature
        rs: (optional) increasing sequence of rs
        Must input either rmax or rs.
        
        Returns:
        rs: sequence of radii in range [0, rmax]. 
            If not initially given: r_j is estimated Riemannian distance from x_i to its jth nearest neighbor.
            If given as input: returns given rs
        ball_volumes: sequence of estimated geodesic ball volumes, for each radius r in rs
        '''
        assert (rmax is not None) or (rs is not None)
        if rmax is not None:
            rs, nbrs = self.nbr_distances(i, rmax)
            N_rs = [(j+2) for j in range(len(rs))] # N_r = number of points in ball of radius r, for r in rs
        else:
            nbr_dists, nbrs = self.nbr_distances(i, rs[-1])
            num_nbrs = len(nbr_dists)
            N_rs = []
            k = 0
            for r in rs:
                while k < num_nbrs and nbr_dists[k] <= r:
                    k += 1
                N_rs.append(k+1) # k neighbors within B_r, plus the center point
        
        # Calculate average ball density for every ball in the sequence
        density = self.get_density()
        center_little_ball_vol = self.Vn*(.5*self.nearest_nbr_dist[i])**self.n
        num = density[i]*center_little_ball_vol
        denom = center_little_ball_vol
        avg_ball_density = []
        if rmax is not None:
            for nbr_idx in nbrs:
                little_ball_vol = self.Vn*(.5*self.nearest_nbr_dist[nbr_idx])**self.n
                num += density[nbr_idx]*little_ball_vol
                denom += little_ball_vol
                avg_ball_density.append(num/denom)
        else:
            k = 0
            for r in rs:
                if k < num_nbrs and nbr_dists[k] <= r:
                    nbr_idx = nbrs[k]
                    little_ball_vol = self.Vn*(.5*self.nearest_nbr_dist[nbr_idx])**self.n
                    num += density[nbr_idx]*little_ball_vol
                    denom += little_ball_vol
                    k += 1
                avg_ball_density.append(num/denom)
        
        # Calculate estimated ball volumes via MLE formula
        ball_volumes = [N_rs[j]/(self.N*avg_ball_density[j]) for j in range(len(rs))]
            
        return rs, ball_volumes
       
    def compute_ball_ratios(self, i):
        # only to be called by compute_ball_ratio_seqs
        _, ball_ratios = self.ball_ratios(i, rs = self.rs)
        return ball_ratios
    
    def compute_ball_ratio_seqs(self, rmax):
        rs, ball_ratios = self.ball_ratios(0, rmax)
        self.rs = rs
        ball_ratio_seqs = [None for i in range(self.N)]
        ball_ratio_seqs[0] = ball_ratios
        
        with mp.Pool(mp.cpu_count()) as p:
            ball_ratio_seqs[1:] = p.map(self.compute_ball_ratios, np.arange(1, self.N))
        
        self.ball_ratio_seqs = ball_ratio_seqs
        return ball_ratio_seqs
        
    def compute_Rdist(X, n_nbrs = 20):
        # n_nbrs: integer. Parameter to pass to isomap. (Number of neighbors in nearest neighbor graph)
        iso = Isomap(n_neighbors = n_nbrs, n_jobs = -1)
        iso.fit(X)
        Rdist = iso.dist_matrix_
        return Rdist
    
    def compute_density(n, X, kernel = None):
        kde = KDE(n, X, kernel)
        density = kde.density()
        return density
    
    def estimate_all(self, rmax, k = 0):
        # k: number of neighbors to average ball_ratios over
        self.k = k
        
        #self.compute_ball_ratio_seqs(rmax)
        rs, ball_ratios = self.ball_ratios(0, rmax)
        self.rs = rs
        
        self.rmax = rmax
        with mp.Pool(mp.cpu_count()) as p:
            Cs = p.map(self.fit_quad_coeff_helper, [i for i in range(self.N)])
        Ss = [-6*(self.n + 2)*C for C in Cs]
        return Ss
        
    def fit_quad_coeff(self, i, rmin = None, rmax = None, rs = None, version = 1):
        '''
        Parameters
        ----------
        i : integer
            index of observation in X (a row)
        rmin : nonnegative real number, optional
            Radius at which to start the sequence of rs, if rs isn't given. The default is None.
        rmax : positive real number, optional
            Max radius to consider in the sequence of rs, if rs isn't given. The default is None.
        rs : increasing sequence of nonnegative real numbers, optional
        version : {1, 2}, optional
            version 1 corresponds to Eq (10) in the overleaf. version 2 is Eq (11). The default is 1.

        Returns
        -------
        C : real number
            The quadratic coefficient of a polynomial of form 1 + C*r^2 that we fit to the data (r_i, y_i = estimated ball ratio at radius r_i)

        '''
        rs, ball_ratios = self.ball_ratios(i, rmax = rmax, rs = rs)
        if rmin is not None:
            ball_ratios = ball_ratios[rs > rmin]
            rs = rs[rs > rmin]

        if version == 1:
            numerator = sum(np.array([(ball_ratios[i] - 1)*r**2 for i, r in enumerate(rs)]))
            denom = sum(np.array([r**4 for r in rs]))
            C = numerator/denom
        else:
            rs = np.append(rs, 0) # so that r[-1] = 0. need this for the rs[i] - rs[i-1] term below.
            numerator = sum(np.array([(r**2)*(ball_ratios[i] - 1)*(r - rs[i-1]) for i, r in enumerate(rs[:-1])]))
            denom = rs[-2]**5/5
            C = numerator/denom
        return C
    
    def fit_quad_coeff_helper(self, i):
        _, nbrs = self.nbr_distances(i, self.rmax)
        k = self.k
        k_nbrs = nbrs[:k]
        _, i_ball_ratios = self.ball_ratios(i, rs = self.rs)
        ball_ratio_sums = i_ball_ratios
        for nbr in k_nbrs:
            _, nbr_ball_ratios = self.ball_ratios(nbr, rs = self.rs)
            for j in range(len(self.rs)):
                ball_ratio_sums[j] = ball_ratio_sums[j] + nbr_ball_ratios[j]
        ball_ratio_avgs = [ball_sum/(k+1) for ball_sum in ball_ratio_sums]       
        numerator = sum(np.array([(ball_ratio_avgs[j] - 1)*r**2 for j, r in enumerate(self.rs)]))
        denom = sum(np.array([r**4 for r in self.rs]))
        C = numerator/denom
        return C
    
    def get_ball_ratio_seqs(self, rmax = None):
        if self.ball_ratio_seqs is None:
            self.compute_ball_ratio_seqs(rmax)
        return self.ball_ratio_seqs
    
    def get_density(self):
        if self.density is None:
            self.density = scalar_curvature_est.compute_density(self.n, self.X, self.kernel)
        return self.density
    
    def get_Rdist(self):
        if self.Rdist is None:
            self.Rdist = scalar_curvature_est.compute_Rdist(self.X, self.n_nbrs)
        return self.Rdist
    
    def nbr_distances(self, i, rmax):
        '''
        i: index of observation in X (i.e., index of a row of X) at which we're estimating scalar curvature
        rmax: positive real number. scale at which we're computing scalar curvature
        
        Returns
            nbr_indices: sorted (ascending order by distance to X[i, :]) list of indices of neighbors that are within rmax of X[i, :]
            distances: sorted (ascending order) list of Riemannian distances from X[i, :] to its neighbors that are within rmax of X[i, :]
        '''
        Rdist = self.get_Rdist()
        distances = Rdist[i, :]
        
        nbr_indices = np.argsort(distances)[1:] # get nbr indices in order (sorted by distance from i), and remove index i
        distances = distances[nbr_indices]
        close_enough = (distances <= rmax)
        distances = distances[close_enough]
        nbr_indices = nbr_indices[close_enough]
        return distances, nbr_indices
    
    def quad_coeff_errs(self, i, rmax, version = 1, l2norm = True):
        '''
        i: index of observation in X (a row) at which we're estimating scalar curvature
        rmax: positive real number.
        version : {1, 2}, optional
            version 1 corresponds to Eq (10) in the overleaf. version 2 is Eq (11). The default is 1.  
        '''
        rs, ball_ratios = self.ball_ratios(i, rmax)
        if l2norm:
            numerator_terms = np.array([(ball_ratios[i] - 1)*r**2 for i, r in enumerate(rs)])
            denom_terms = np.array([r**4 for r in rs])
            num_sum = 0
            denom_sum = 0
            Cs = []
            for i in range(len(rs)):
                num_sum += numerator_terms[i]
                denom_sum += denom_terms[i]
                C = num_sum/denom_sum
                Cs.append(C)
            errs = []
            err1 = 0
            err2 = 0 
            err3 = 0
            for i, r in enumerate(rs):
                err1 += r**4
                err2 += (r**2)*(1 - ball_ratios[i])
                err3 += (1 - ball_ratios[i])**2
                sq_err = (Cs[i]**2)*err1 + 2*Cs[i]*err2 + err3
                avg_err = math.sqrt(max(0, sq_err))/r # theoretically, sq_err is always positive. because of floating point errors, it can be negative (but very very small, on the order of 10^(-17))
                errs.append(avg_err)
        else:
            rs = np.append(rs, 0)
            numerator_terms = np.array([(r**2)*(ball_ratios[i] - 1)*(r - rs[i-1]) for i, r in enumerate(rs[:-1])])
            denom = (rs[-2]**5)/5
            num_sum = 0
            Cs = []
            for i, num in enumerate(numerator_terms):
                num_sum += num
                C = num_sum/denom
                Cs.append(C)
            err1 = 0
            err2 = 0
            err3 = 0
            errs = []
            for i, r in enumerate(rs[:-1]):
                err1 += (r**4)*(r - rs[i-1])
                err2 += (r**2)*(1 - ball_ratios[i])*(r - rs[i-1])
                err3 += ((1 - ball_ratios[i])**2)*(r - rs[i-1])
                sq_err = (Cs[i]**2)*err1 + 2*Cs[i]*err2 + err3
                avg_err = math.sqrt(max(0, sq_err))/r # theoretically, sq_err is always positive. because of floating point errors, it can be negative (but very very small, on the order of 10^(-17))
                errs.append(avg_err)
        return rs, Cs, errs