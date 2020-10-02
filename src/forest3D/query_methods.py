import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
#import gurobipy
import math
from sklearn.neighbors import KDTree
from scipy import stats
import sys
from sklearn.cluster import KMeans


def query_random(poolSize,budget):

    return np.random.permutation(poolSize)[:budget]


def query_uncertainty(probs,budget):

    return np.argsort(probs)[:budget]


def query_marginal_uncertainty(probs,budget):
    # probs is the difference between most certain two classes

    return np.argsort(probs)[:budget]



def query_densityWeightedUncertainty_fast(codes_U, probs_U, budget, diversity=0, frontier=0, density=0.1):
    # fast because it doesnt use gurobi (therefore can run on gpu)

    nSamples = np.shape(codes_U)[0]

    # pre-compute pairwise distance matrix
    rows, cols = np.indices((nSamples, nSamples))
    locs = rows <= cols
    dist_mat = euclidean_distances(codes_U, codes_U)
    dist_mat2 = dist_mat.copy()
    dist_mat[locs] = 0
    spread = np.mean(dist_mat2, axis=0)

    spread *= density
    total = spread + probs_U
    solution = np.argsort(total)[:budget]

    return solution



def query_neighborhoodSearch(codes,probs,budget,numIters = 100,returnAssigns=False,power=1):

    # pre-compute pairwise distance matrix
    probs = (1 - probs) ** power
    nSamples = len(probs)
    rows, cols = np.indices((nSamples, nSamples))
    locs = rows <= cols
    dist_mat = euclidean_distances(codes, codes)

    cost_mat = dist_mat * np.tile(probs, (nSamples, 1)).T


    # randomly assign p candidate locations
    y = np.zeros((nSamples))
    y[np.random.permutation(nSamples)[:budget]] = 1

    for j in range(numIters):

        # assign nodes to nearest candidate
        y_dist = dist_mat[y == 1, :]
        assign = np.argmin(y_dist, axis=0)

        new_candidiates = []
        for i in range(budget):

            # method 1: find sample (centroid) which minimises uncertainty-weighted distance to all points per assignment
            xv, yv = np.meshgrid(np.where(assign == i), np.where(assign == i))
            sub_mat = np.transpose( np.reshape(cost_mat[np.ravel(xv), np.ravel(yv)], (np.sum(assign == i), np.sum(assign == i))) )
            idx = np.argmin(np.sum(sub_mat, axis=0))
            new_candidiates.append(np.where(assign == i)[0][idx])

            # method 2: mean of codes weighted by uncertainty
            # mean = np.sum( codes[assign==i,:]*(1-probs[assign==i])[:,np.newaxis] , axis=0 ) / np.sum(1-probs[assign==i])
            # new_candidiates.append( np.where(assign==i)[0][[np.argmin( euclidean_distances(mean[np.newaxis, :], codes[assign==i]) )]][0] )


        # assign p new candidate locations
        y = np.zeros((nSamples))
        y[new_candidiates] = 1

    if returnAssigns:
        return np.where(y == 1)[0],assign
    else:
        return np.where(y==1)[0]



def query_neighborhoodSearch_rep(codes,probs,budget,numIters = 100,numReps=20,returnAssigns=False,power=1):
    '''

    :param codes:
    :param probs: this must be certainty (i.e. 1.0 is certain/high probability, 0.0 is uncertain/low probability
    :param budget:
    :param numIters:
    :param numReps:
    :param returnAssigns:
    :return:
    '''

    # pre-compute pairwise distance matrix
    probs = (1 - probs)**power
    nSamples = len(probs)
    rows, cols = np.indices((nSamples, nSamples))
    locs = rows <= cols
    dist_mat = euclidean_distances(codes, codes)

    cost_mat = dist_mat * np.tile(probs, (nSamples, 1)).T

    best_cost = float('inf')
    best_idx = []
    best_assign = []

    for k in range(numReps):

        rep_cost = 0

        # randomly assign p candidate locations
        y = np.zeros((nSamples))
        y[np.random.permutation(nSamples)[:budget]] = 1

        for j in range(numIters):

            # assign nodes to nearest candidate
            y_dist = dist_mat[y == 1, :]
            assign = np.argmin(y_dist, axis=0)

            new_candidiates = []
            for i in range(budget):

                # method 1: find sample (centroid) which minimises uncertainty-weighted distance to all points per assignment
                xv, yv = np.meshgrid(np.where(assign == i), np.where(assign == i))
                sub_mat = np.transpose( np.reshape(cost_mat[np.ravel(xv), np.ravel(yv)], (np.sum(assign == i), np.sum(assign == i))) )
                idx = np.argmin(np.sum(sub_mat, axis=0))
                rep_cost+= np.min(np.sum(sub_mat, axis=0))
                new_candidiates.append(np.where(assign == i)[0][idx])



            # assign p new candidate locations
            y = np.zeros((nSamples))
            y[new_candidiates] = 1

        if rep_cost<best_cost:
            best_cost = rep_cost
            best_idx = np.where(y == 1)[0]
            best_assign = assign


    if returnAssigns:
        return best_idx,best_assign
    else:
        return best_idx





def cluster_exemplers(codes,budget):

    kmeans = KMeans(budget,n_init=100)
    kmeans.fit(codes)

    dist_mat = euclidean_distances(codes, kmeans.cluster_centers_)

    p = range(budget)

    querries = []

    for i in range(budget):
        min_score = float('inf')
        min_arg = None
        min_j = None
        for j in p:
            idx = np.argmin(dist_mat[:,j])
            if dist_mat[idx,j] < min_score:
                min_score = dist_mat[idx,j]
                min_arg = idx
                min_j = j
        querries.append(min_arg)
        dist_mat[min_arg,:] = float('inf')
        p.remove(min_j)

    return querries






if 'gurobipy' in sys.modules:


    def query_proposed(codes_L, labels_L, codes_U, probs_U, budget, diversity=0.03, frontier=1.0, density=0.1, uncert=1.0, time_limit=5.0):

        querrier = latent_querries(codes_L, labels_L, codes_U, probs_U)
        solution = querrier.solve(budget, time_limit=time_limit, nearest_neighbours=5, alpha=diversity, beta=frontier,
                                  gamma=density, delta=uncert)

        return solution  # [37, 91, 110, 163, 218, 501, 636, 672, 763, 998]



    def query_densityWeightedUncertainty(codes_U, probs_U, budget, diversity=0, frontier=0, density=0.1):
        # only uses codes_U and probs_U. But need to input L to use latent querries class

        querrier = latent_querries(None, None, codes_U, probs_U)
        solution = querrier.solve(budget, time_limit=5.0, nearest_neighbours=5, alpha=diversity, beta=frontier,
                                  gamma=density)

        return solution  # [37, 91, 110, 163, 218, 501, 636, 672, 763, 998]



    def query_latent(codes_L, labels_L, codes_U, probs_U, budget, diversity=0.03, frontier=0, density=0.1, time_limit=5.0):

        querrier = latent_querries(codes_L, labels_L, codes_U, probs_U)
        solution = querrier.solve(budget, time_limit=time_limit, nearest_neighbours=5, alpha=diversity, beta=frontier,
                                  gamma=density)

        return solution


    class latent_querries():

        def __init__(self,test_code_l, test_labels_l, test_code_u, preds_prob_u):

            self.test_code_l = test_code_l
            self.test_labels_l = test_labels_l
            self.test_code_u = test_code_u
            self.preds_prob_u = preds_prob_u

            self.nSamples = np.shape(test_code_u)[0]

            # pre-compute pairwise distance matrix
            rows, cols = np.indices((self.nSamples, self.nSamples))
            locs = rows<=cols
            self.dist_mat = euclidean_distances(test_code_u, test_code_u)
            dist_mat2 = self.dist_mat.copy()
            self.dist_mat[ locs ] = 0


            # pre-compute density or similarity score for each samples
            self.spread = np.mean(dist_mat2,axis=0)

        def solve(self,budget,time_limit=5.0,nearest_neighbours=5, alpha=0.03, beta=1.0, gamma=0.1, delta=1.0):

            # pre-compute number of nearest dominant label for each sample
            if self.test_code_l is not None:
                X_train = KDTree(self.test_code_l)
                _,qInd = X_train.query(self.test_code_u,nearest_neighbours)
                nearest_labels = np.argmax(self.test_labels_l,axis=1)[qInd]
                _,count = stats.mode(nearest_labels, axis=1)
                fl_score = count[:,0]/np.float(nearest_neighbours) # frontier likelihood score (count normalised by num nearest neighbours)
            else:
                fl_score = np.zeros((self.nSamples))



            model = gurobipy.Model('latent_space')

            # binary variable for each sample index
            var_samples = model.addVars(range(self.nSamples), vtype=gurobipy.GRB.BINARY, name='samples')

            # constrain so that number of 1's are equal to the budget
            model.addConstr( (var_samples.sum() == budget), name='budget_constr' )


            # compute sum of pairwise distances for sample indexes with 1 (i.e. diversity sum)
            sol_dist_mat = []
            [sol_dist_mat.append([]) for i in range(self.nSamples)]
            [sol_dist_mat[i].append([]) for j in range(self.nSamples) for i in range(self.nSamples)]
            for i in range(self.nSamples):
                for j in range(self.nSamples):
                    sol_dist_mat[i][j] =  var_samples[i] * self.dist_mat[i,j]

            sum_dist_mat = []
            for i in range(self.nSamples):
                sum_dist_mat.append( var_samples.prod(sol_dist_mat[i]) )
            diversity = np.sum( sum_dist_mat )

            # normalise by dividing by number of pairs (hence, it is the mean pairwise distance for the selected samples)
            nPairs = math.factorial(budget)/(math.factorial(2)*(math.factorial(budget-2))) # budget C 2
            diversity = diversity / nPairs

            # compute the mean uncertainty of the selected samples
            uncertainty = (-1*var_samples.prod( list( self.preds_prob_u ) ))/budget

            # compute the density score of the selected samples
            density = (-1*var_samples.prod( list( self.spread ) ))/budget

            #
            frontier_likelihood = (-1*var_samples.prod( list( fl_score ) ))/budget

            # objective function is the alpha weighted combination of mean diversity and mean uncertainty
            obj = alpha*diversity + delta*uncertainty + beta*frontier_likelihood + gamma*density

            model.setObjective(obj)

            # want to maximise objective
            model.ModelSense = gurobipy.GRB.MAXIMIZE

            # run optimisation
            model.Params.timeLimit = time_limit*60
            model.Params.mipgap = 0.01
            model.optimize()

            print('Total value = $' + str(model.ObjVal))

            # print optimal indexes for maximum objective
            solution = np.where( np.array(model.X) > 0.5 )

            return list(solution[0])

