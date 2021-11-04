import os

import numpy as np
import scipy.stats
import torch
# delta = partial(np.linalg.norm, )
from sklearn.metrics import pairwise_distances
from torch.autograd import Variable
from torch.nn import functional as F
from analysis.private_knn import PrivateKnn

delta = np.linalg.norm
import abc
import numpy
# from gurobipy import Model, LinExpr, UB, GRB
import numpy.matlib
import time
from analysis import analyze_multiclass_gnmax


class SamplingMethod(object):

    @abc.abstractmethod
    def __init__(self):
        __metaclass__ = abc.ABCMeta

    def flatten_X(self, X):
        shape = X.shape
        flat_X = X
        if len(shape) > 2:
            flat_X = np.reshape(X, (shape[0], np.product(shape[1:])))
        return flat_X

    @abc.abstractmethod
    def select_batch_(self):
        return

    def select_batch(self, **kwargs):
        return self.select_batch_(**kwargs)

    def to_dict(self):
        return None


class kCenterGreedy(SamplingMethod):

    def __init__(self, metric='euclidean'):
        super().__init__()
        self.name = 'kcenter'
        self.metric = metric
        self.min_distances = None
        self.already_selected = []

    def update_distances(self, features, cluster_centers, only_new=True,
                         reset_dist=False):
        """Update min distances given cluster centers.
        Args:
          features: features (projection) from model
          cluster_centers: indices of cluster centers
          only_new: only calculate distance for newly selected points and update
            min_distances.
          rest_dist: whether to reset min_distances.
        """

        if reset_dist:
            self.min_distances = None
        if only_new:
            cluster_centers = [d for d in cluster_centers
                               if d not in self.already_selected]
        if cluster_centers:
            # Update min_distances for all examples given new cluster center.
            x = features[cluster_centers]
            dist = pairwise_distances(features.detach().numpy(),
                                      x.detach().numpy(), metric=self.metric)

            if self.min_distances is None:
                self.min_distances = np.min(dist, axis=1).reshape(-1, 1)
            else:
                self.min_distances = np.minimum(self.min_distances, dist)

    def select_batch_(self, pool, model, already_selected, N, **kwargs):
        """
        Diversity promoting active learning method that greedily forms a batch
        to minimize the maximum distance to a cluster center among all unlabeled
        datapoints.
        Args:
          pool: tuple of (X, Y)
          model: model with scikit-like API with decision_function implemented
          already_selected: index of datapoints already selected
          N: batch size
        Returns:
          indices of points selected to minimize distance to cluster centers
        """

        try:
            # Assumes that the transform function takes in original data and not
            # flattened data.
            print('Getting transformed features...')
            features = model.forward(pool[0].float())
            print('Calculating distances...')
            self.update_distances(features, already_selected, only_new=False,
                                  reset_dist=True)
        except Exception as e:
            print(f"error: {e}")
            print('Using flat_X as features.')
            self.update_distances(features, already_selected, only_new=True,
                                  reset_dist=False)

        new_batch = []

        for _ in range(N):
            if self.already_selected is None:
                # Initialize centers with a randomly selected datapoint
                ind = np.random.choice(np.arange(pool[0].shape[0]))
            else:
                ind = np.argmax(self.min_distances)
            # New examples should not be in already selected since those points
            # should have min_distance of zero to a cluster center.
            assert ind not in already_selected

            self.update_distances(features, [ind], only_new=True,
                                  reset_dist=False)
            new_batch.append(ind)
        print(
            f"Maximum distance from cluster centers is {max(self.min_distances)}.")
        self.already_selected = already_selected

        return new_batch


# def solve_fac_loc(xx, yy, subset, n, budget):
#   model = Model("k-center")
#   x = {}
#   y = {}
#   z = {}
#   for i in range(n):
#     # z_i: is a loss
#     z[i] = model.addVar(obj=1, ub=0.0, vtype="B", name="z_{}".format(i))
#
#   m = len(xx)
#   for i in range(m):
#     _x = xx[i]
#     _y = yy[i]
#     # y_i = 1 means i is facility, 0 means it is not
#     if _y not in y:
#       if _y in subset:
#         y[_y] = model.addVar(obj=0, ub=1.0, lb=1.0, vtype="B", name="y_{}".format(_y))
#       else:
#         y[_y] = model.addVar(obj=0, vtype="B", name="y_{}".format(_y))
#     # if not _x == _y:
#     x[_x, _y] = model.addVar(obj=0, vtype="B", name="x_{},{}".format(_x, _y))
#   model.update()
#
#   coef = [1 for j in range(n)]
#   var = [y[j] for j in range(n)]
#   model.addConstr(LinExpr(coef, var), "=", rhs=budget + len(subset), name="k_center")
#
#   for i in range(m):
#     _x = xx[i]
#     _y = yy[i]
#     # if not _x == _y:
#     model.addConstr(x[_x, _y], "<", y[_y], name="Strong_{},{}".format(_x, _y))
#
#   yyy = {}
#   for v in range(m):
#     _x = xx[v]
#     _y = yy[v]
#     if _x not in yyy:
#       yyy[_x] = []
#     if _y not in yyy[_x]:
#       yyy[_x].append(_y)
#
#   for _x in yyy:
#     coef = []
#     var = []
#     for _y in yyy[_x]:
#       # if not _x==_y:
#       coef.append(1)
#       var.append(x[_x, _y])
#     coef.append(1)
#     var.append(z[_x])
#     model.addConstr(LinExpr(coef, var), "=", 1, name="Assign{}".format(_x))
#   model.__data = x, y, z
#   return model


def greedy_k_center(model, pool, already_selected, batch_size):
    # note pool should have all points in a tuple of (X, Y)
    # already selected are the indices
    # this returns the indices o the selected samples
    selecter = kCenterGreedy()
    return selecter.select_batch_(pool, model, already_selected, batch_size)


def robust_k_center(x, y, z):
    budget = 10000

    start = time.clock()
    num_images = x.shape[0]
    dist_mat = numpy.matmul(x, x.transpose())

    sq = numpy.array(dist_mat.diagonal()).reshape(num_images, 1)
    dist_mat *= -2
    dist_mat += sq
    dist_mat += sq.transpose()

    elapsed = time.clock() - start
    print(f"Time spent in (distance computation) is: {elapsed}")

    num_images = 50000

    # We need to get k centers start with greedy solution
    budget = 10000
    subset = [i for i in range(1)]

    ub = UB
    lb = ub / 2.0
    max_dist = ub

    _x, _y = numpy.where(dist_mat <= max_dist)
    _d = dist_mat[_x, _y]
    subset = [i for i in range(1)]
    model = solve_fac_loc(_x, _y, subset, num_images, budget)
    # model.setParam( 'OutputFlag', False )
    x, y, z = model.__data
    delta = 1e-7
    while ub - lb > delta:
        print("State", ub, lb)
        cur_r = (ub + lb) / 2.0
        viol = numpy.where(_d > cur_r)
        new_max_d = numpy.min(_d[_d >= cur_r])
        new_min_d = numpy.max(_d[_d <= cur_r])
        print("If it succeeds, new max is:", new_max_d, new_min_d)
        for v in viol[0]:
            x[_x[v], _y[v]].UB = 0

        model.update()
        r = model.optimize()
        if model.getAttr(GRB.Attr.Status) == GRB.INFEASIBLE:
            failed = True
            print("Infeasible")
        elif sum([z[i].X for i in range(len(z))]) > 0:
            failed = True
            print("Failed")
        else:
            failed = False
        if failed:
            lb = max(cur_r, new_max_d)
            # failed so put edges back
            for v in viol[0]:
                x[_x[v], _y[v]].UB = 1
        else:
            print("sol found", cur_r, lb, ub)
            ub = min(cur_r, new_min_d)
            model.write("s_{}_solution_{}.sol".format(budget, cur_r))


# def k_center_greedy(pool, s0, budget):
#   s = s0
#   new_points = []
#   init_size = len(s0)
#   s0 = set(s0)
#   while len(s) < init_size + budget:
#     max_dist, index = 0, -1
#     for i in range(len(pool)):
#       if i in s:
#         continue
#       min_dist = 1000000000
#       for j in s:
#         min_dist = min(min_dist, delta(pool[i], pool[j]))
#       if min_dist > max_dist:
#         max_dist = min_dist
#         index = i
#     if index < 0:
#       raise ValueError('index should not have been -1, error ')
#     s.append(index)
#     new_points.append(index)
#   return s, new_points
#
# def feasible():
#   pass


# def robust_k_center(pool, s0, budget, bound):
#   greedy_init, greedy_points = k_center_greedy(pool, s0, budget)
#   d2_opt = 0
#   for i in range(len(pool)):
#     min_dist = 10000000
#     for j in greedy_init:
#       min_dist = min(min_dist, delta(pool[i], pool[j]))
#     d2_opt = max(d2_opt, min_dist)
#   lb = d2_opt / 2
#   ub = d2_opt
#   while lb < ub:
#     midpoint = (lb + ub) / 2
#     if feasible(budget, s0, midpoint, bound):
#       new_ub = 0
#       for i in range(len(pool)):
#         for j in greedy_init:
#           dist = delta(pool[i], pool[j])
#           if dist > new_ub and dist < midpoint:
#             new_ub = dist
#       ub = new_ub
#     else:
#       new_lb = 10000000000
#       for i in range(len(pool)):
#         for j in greedy_init:
#           dist = delta(pool[i], pool[j])
#           if dist < new_lb and dist > midpoint:
#             new_lb = dist
#       lb = new_lb

def get_model_name(model):
    if getattr(model, 'module', '') == '':
        return model.name
    else:
        return model.module.name


def compute_utility_scores_entropy(model, dataloader, args):
    """Assign a utility score to each data sample from the unlabeled dataset."""
    with torch.no_grad():
        # Entropy value as a proxy for utility.
        entropy = []
        for data, _ in dataloader:
            if args.cuda:
                data = data.cuda()
            output = model(data)
            prob = F.softmax(output, dim=1).cpu().numpy()
            entropy.append(scipy.stats.entropy(prob, axis=1))
        entropy = np.concatenate(entropy, axis=0)
        # Maximum entropy is achieved when the distribution is uniform.
        entropy_max = np.log(args.num_classes)
        # Sanity checks
        try:
            assert len(entropy.shape) == 1 and entropy.shape[0] == len(
                dataloader.dataset)
            assert np.all(entropy <= entropy_max) and np.all(0 <= entropy)
        except AssertionError:
            # change nan to 0 and try again
            entropy[np.isnan(entropy)] = 0
            assert len(entropy.shape) == 1 and entropy.shape[0] == len(
                dataloader.dataset)
            assert np.all(entropy <= entropy_max) and np.all(0 <= entropy)
            print("There are NaNs in the utlity scores, reset to 0")
        # Normalize utility scores to [0, 1]
        utility = entropy / entropy_max
        # Save utility scores
        filename = "{}-utility-scores-(mode:entropy)".format(
            get_model_name(model))
        filepath = os.path.join(args.ensemble_model_path, filename)
        np.save(filepath, utility)
        return utility


def get_train_representations(model, trainloader, args):
    """
    Compute the train representations for the training set.

    :param model: ML model
    :param trainloader: data loader for training set
    :param args: the parameters for the program

    :return: training representations and their targets
    """
    train_represent = []
    train_labels = []
    with torch.no_grad():
        for batch_id, (data, target) in enumerate(trainloader):
            if args.cuda:
                data = data.cuda()
            outputs = model(data)
            outputs = F.log_softmax(outputs, dim=-1)
            outputs = outputs.cpu().numpy()
            train_represent.append(outputs)
            train_labels.append(target.cpu().numpy())
    train_represent = np.concatenate(train_represent, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    return train_represent, train_labels


def get_votes_for_pate_knn(model, unlabeled_loader, train_represent,
                           train_labels, args):
    """

    :param model: the model to be used
    :param unlabeled_loader: data points to be labeled - for which we compute
        the score
    :param train_represent: last layer representation for the teachers
    :param train_labels: labels for the teachers
    :param args: the program parameters

    :return: votes for each data point

    """

    # num_teachers: number of k nearest neighbors acting as teachers
    num_teachers = args.num_teachers_private_knn

    with torch.no_grad():
        # Privacy cost as a proxy for utility.
        votes = []
        targets = []
        predictions = []
        for data, target in unlabeled_loader:
            if args.cuda:
                data = data.cuda()
            outputs = model(data)
            outputs = F.log_softmax(outputs, dim=-1)
            outputs = outputs.cpu().numpy()
            targets.append(target.cpu().numpy())
            predictions.append(np.argmax(outputs, axis=-1))
            for output in outputs:
                dis = np.linalg.norm(train_represent - output, axis=-1)
                k_index = np.argpartition(dis, kth=num_teachers)[:num_teachers]
                teachers_preds = np.array(train_labels[k_index], dtype=np.int32)
                label_count = np.bincount(
                    teachers_preds, minlength=args.num_classes)
                votes.append(label_count)
        # targets = np.concatenate(targets, axis=-1)
        # predictions = np.concatenate(predictions, axis=-1)
        # correct = (predictions == targets).sum()
        # len_targets = len(targets)
        # accuracy = correct / len_targets
        # print(f'prediction accuracy for {len_targets} targets: {accuracy}')
        # votes_predict = np.argmax(votes, axis=-1)
        # votes_correct = (votes_predict == targets).sum()
        # votes_acc = votes_correct / len_targets
        # print(f'voting accuracy for {len_targets} targets: {votes_acc}')
    votes = np.stack(votes)
    return votes


def compute_utility_scores_pate_knn(
        model, unlabeled_loader, args, trainloader, train_represent=None,
        train_labels=None):
    """Assign a utility score to each data sample from the unlabeled dataset.
    Either trainloader or train_represent has to be provided.

    :param model: the model to be used
    :param unlabeled_loader: data points to be labeled - for which we compute
        the score
    :param args: the program parameters
    :param trainloader: the data loader for the training set
    :param train_represent: last layer representation for the teachers
    :param train_labels: labels for the teachers

    :return: utility score based on the privacy budget for each point in the
    dataset unlabeled_loader
    """
    if train_represent is None:
        assert trainloader is not None
        train_represent, train_labels = get_train_representations(
            model=model, trainloader=trainloader, args=args)

    votes = get_votes_for_pate_knn(
        model=model, train_labels=train_labels, train_represent=train_represent,
        args=args, unlabeled_loader=unlabeled_loader
    )

    max_num_query, dp_eps, _, _, _ = analyze_multiclass_gnmax(
        votes=votes,
        threshold=0,
        sigma_threshold=0,
        sigma_gnmax=args.sigma_gnmax_private_knn,
        budget=np.inf,
        delta=args.delta,
        show_dp_budget=args.show_dp_budget,
        args=args)
    # Make sure we compute the privacy loss for all queries.
    assert max_num_query == len(votes)
    privacy_cost = dp_eps
    # Save utility scores
    filename = "{}-utility-scores-mode-privacy".format(
        get_model_name(model))
    filepath = os.path.join(args.ensemble_model_path, filename)
    np.save(filepath, privacy_cost)
    return privacy_cost


class PateKNN:
    """
    Compute the privacy cost.
    """

    def __init__(self, model, trainloader, args):
        """

        Args:
            model: the victim model.
            trainloader: the data loader for the training data.
            args: the program parameters.
        """
        self.model = model
        self.args = args

        # Extract the last layer representation of the training points and their
        # ground-truth labels.
        self.train_represent, self.train_labels = get_train_representations(
            model=model, trainloader=trainloader, args=args)

        self.private_knn = PrivateKnn(
            delta=args.delta, sigma_gnmax=args.sigma_gnmax_private_knn,
            apply_data_independent_bound=args.apply_data_independent_bound)

    def compute_privacy_cost(self, unlabeled_loader):
        """

        Args:
            unlabeled_loader: data loader for new queries.

        Returns:
            The total privacy cost incurred by all the queries seen so far.

        """
        votes = get_votes_for_pate_knn(
            model=self.model, train_labels=self.train_labels,
            train_represent=self.train_represent, args=self.args,
            unlabeled_loader=unlabeled_loader
        )

        dp_eps = self.private_knn.add_privacy_cost(votes=votes)

        return dp_eps


def compute_utility_scores_gap(model, dataloader, args):
    """Assign a utility score to each data sample from the unlabeled dataset."""
    with torch.no_grad():
        # Gap between the probabilities of the two most probable classes as a proxy for utility.
        gap = []
        for data, _ in dataloader:
            if args.cuda:
                data = data.cuda()
            output = model(data)
            sorted_output = output.sort(dim=-1, descending=True)[0]
            prob = F.softmax(sorted_output[:, :2], dim=1).cpu().numpy()
            gap.append(prob[:, 0] - prob[:, 1])
        gap = np.concatenate(gap, axis=0)
        # Sanity checks
        try:
            assert len(gap.shape) == 1 and gap.shape[0] == len(
                dataloader.dataset)
            assert np.all(gap <= 1) and np.all(
                0 <= gap), f"gaps: {gap.tolist()}"
        except AssertionError:
            # change nan to 0 and try again
            gap[np.isnan(gap)] = 0
            assert len(gap.shape) == 1 and gap.shape[0] == len(
                dataloader.dataset)
            assert np.all(gap <= 1) and np.all(
                0 <= gap), f"gaps: {gap.tolist()}"
            print("There are NaNs in the utlity scores, reset to 0")
        # Convert gap values into utility scores
        utility = 1 - gap
        # Save utility scores
        filename = "{}-utility-scores-(mode:gap)".format(get_model_name(model))
        filepath = os.path.join(args.ensemble_model_path, filename)
        np.save(filepath, utility)
        return utility


def compute_utility_scores_greedy(model, dataloader, args):
    model.cpu()
    with torch.no_grad():
        samples = []
        for data, _ in dataloader:
            data = Variable(data)
            samples.append(data)
        samples = torch.cat(samples, dim=0)
        indices = greedy_k_center(model, (samples, None), [],
                                  len(dataloader.dataset))
        try:
            assert len(indices) == len(dataloader.dataset) and len(
                set(indices)) == len(dataloader.dataset)
        except AssertionError:
            print("Assertion Error In Greedy, return all zero utility scores")
            return np.zeros(len(dataloader.dataset))
        indices = np.array(indices)
        utility = np.zeros(len(dataloader.dataset))
        for i in range(len(indices)):
            utility[indices[i]] = (len(dataloader.dataset) - i) / float(
                len(dataloader.dataset))
        # Save utility scores
        filename = "{}-utility-scores-(mode:greedy)".format(
            get_model_name(model))
        filepath = os.path.join(args.ensemble_model_path, filename)
        np.save(filepath, utility)
        if args.cuda:
            model.cuda()
        return utility


def compute_utility_scores_random(model, dataloader, args):
    return np.random.random(len(dataloader.dataset))
