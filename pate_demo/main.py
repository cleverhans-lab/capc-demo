from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import time

import numpy as np
import torch

import analysis
import utils
from active_learning import compute_utility_scores_entropy
from active_learning import compute_utility_scores_gap
from active_learning import compute_utility_scores_greedy
from architectures.densenet_pre import densenetpre
from architectures.resnet_pre import resnetpre
from model_extraction.deepfool import compute_utility_scores_deepfool
from architectures.utils_architectures import pytorch2pickle

# from datasets.chexpert.bin import train_chexpert
from datasets.deprecated.chexpert.bin import train_chexpert
from datasets.deprecated.chexpert.chexpert_utils import get_chexpert_dev_loader
from datasets.utils import get_dataset_full_name
from datasets.utils import set_dataset
from datasets.utils import show_dataset_stats
from datasets.xray.xray_datasets import get_votes_only_for_dataset
from errors import check_perfect_balance_type
from model_extraction.main_model_extraction import run_model_extraction
from models.add_tau_per_model import set_taus
from models.big_ensemble_model import BigEnsembleModel
from models.ensemble_model import EnsembleModel
from models.load_models import load_private_model_by_id
from models.load_models import load_private_models
from models.private_model import get_private_model_by_id
from models.utils_models import get_model_name_by_id
from models.utils_models import model_size
from parameters import get_parameters
from utils import eval_distributed_model
from utils import eval_model
from utils import from_result_to_str
from utils import get_unlabeled_indices
from utils import get_unlabeled_set
from utils import metric
from utils import result
from utils import train_model
from utils import update_summary
from utils import pick_labels_general
from virtual_parties import query_ensemble_model_with_virtual_parties
from model_extraction.adaptive_training import train_model_adaptively


###########################
# ORIGINAL PRIVATE MODELS #
###########################
def train_private_models(args):
    """Train N = num-models private models."""
    start_time = time.time()

    # Checks
    assert 0 <= args.begin_id
    assert args.begin_id < args.end_id
    assert args.end_id <= args.num_models

    # Logs
    filename = "logs-(id:{:d}-{:d})-(num-epochs:{:d}).txt".format(
        args.begin_id + 1, args.end_id, args.num_epochs
    )
    if os.name == "nt":
        filename = "logs-(id_{:d}-{:d})-(num-epochs_{:d}).txt".format(
            args.begin_id + 1, args.end_id, args.num_epochs
        )
    file = open(os.path.join(args.private_model_path, filename), "w+")
    args.log_file = file
    args.save_model_path = args.private_model_path
    utils.augmented_print("##########################################", file)
    utils.augmented_print(
        "Training private models on '{}' dataset!".format(args.dataset), file
    )
    utils.augmented_print(
        "Training private models on '{}' architecture!".format(
            args.architecture), file
    )
    utils.augmented_print(
        "Number of private models: {:d}".format(args.num_models), file
    )
    utils.augmented_print(f"Initial learning rate: {args.lr}.", file)
    utils.augmented_print(
        "Number of epochs for training each model: {:d}".format(
            args.num_epochs), file
    )

    # Data loaders
    if args.dataset_type == "imbalanced":
        all_private_trainloaders = utils.load_private_data_imbalanced(args)
    elif args.dataset_type == "balanced":
        if args.balance_type == "standard":
            all_private_trainloaders = utils.load_private_data(args=args)
        elif args.balance_type == "perfect":
            check_perfect_balance_type(args=args)
            all_private_trainloaders = utils.load_private_data_imbalanced(args)
        else:
            raise Exception(f"Unknown args.balance_type: {args.balance_type}.")
    else:
        raise Exception(f"Unknown dataset type: {args.dataset_type}.")

    evalloader = utils.load_evaluation_dataloader(args)
    # evalloader = utils.load_private_data(args=args)[0]
    print(f"eval dataset: ", evalloader.dataset)

    if args.debug is True:
        # Logs about the eval set
        show_dataset_stats(
            dataset=evalloader.dataset, args=args, file=file,
            dataset_name="eval"
        )

    # Training
    summary = {
        "loss": [],
        "acc": [],
        "balanced_acc": [],
        "auc": [],
    }
    for id in range(args.begin_id, args.end_id):
        utils.augmented_print("##########################################",
                              file)

        # Private model for initial training.
        if args.dataset == "cxpert":
            model = densenetpre()
            print("Loaded densenet121")
        else:
            model = get_private_model_by_id(args=args, id=id)
        # model = densenetpre()
        if args.dataset == "pascal":
            model_state_dict = model.state_dict()
            pretrained_dict34 = torch.load(
                "./architectures/resnet50-19c8e357.pth")
            pretrained_dict_1 = {
                k: v for k, v in pretrained_dict34.items() if
                k in model_state_dict
            }
            model_state_dict.update(pretrained_dict_1)
            model.load_state_dict(model_state_dict)

        trainloader = all_private_trainloaders[id]

        print(f"train dataset for model id: {id}", trainloader.dataset)

        # Logs about the train set
        if args.debug is True:
            show_dataset_stats(
                dataset=trainloader.dataset,
                args=args,
                file=file,
                dataset_name="private train",
            )
        utils.augmented_print("Steps per epoch: {:d}".format(len(trainloader)),
                              file)

        if args.dataset.startswith(
                "chexpert") and not args.architecture.startswith(
            "densenet"
        ):
            devloader = get_chexpert_dev_loader(args=args)
            result, best_model = train_chexpert.run(
                args=args,
                model=model,
                dataloader_train=trainloader,
                dataloader_dev=devloader,
                dataloader_eval=evalloader,
            )
        # elif args.dataset == 'cxpert':
        #     train_cxpert(args=args, model=model, train_loader=trainloader,
        #                  valid_loader=evalloader)
        else:
            train_model(
                args=args, model=model, trainloader=trainloader,
                evalloader=evalloader
            )
            result = eval_distributed_model(
                model=model, dataloader=evalloader, args=args
            )

        model_name = get_model_name_by_id(id=id)
        result["model_name"] = model_name
        result_str = from_result_to_str(result=result, sep=" | ",
                                        inner_sep=": ")
        utils.augmented_print(text=result_str, file=file, flush=True)
        summary = update_summary(summary=summary, result=result)

        # Checkpoint
        state = result
        state["state_dict"] = model.state_dict()
        filename = "checkpoint-{}.pth.tar".format(model_name)
        filepath = os.path.join(args.private_model_path, filename)
        torch.save(state, filepath)

    utils.augmented_print("##########################################", file)

    for key, value in summary.items():
        if len(value) > 0:
            avg_value = np.mean(value)
            utils.augmented_print(
                f"Average {key} of private models: {avg_value}", file)

    end_time = time.time()
    elapsed_time = end_time - start_time
    utils.augmented_print(f"elapsed time: {elapsed_time}\n", file, flush=True)
    utils.augmented_print("##########################################", file)
    file.close()


# def train_private_models(args):
#     """Train N = num-models private models."""
#     start_time = time.time()
#
#     # Checks
#     assert 0 <= args.begin_id
#     assert args.begin_id < args.end_id
#     assert args.end_id <= args.num_models
#
#     # Logs
#     filename = "logs-(id:{:d}-{:d})-(num-epochs:{:d}).txt".format(
#         args.begin_id + 1, args.end_id, args.num_epochs
#     )
#     if os.name == "nt":
#         filename = "logs-(id_{:d}-{:d})-(num-epochs_{:d}).txt".format(
#             args.begin_id + 1, args.end_id, args.num_epochs
#         )
#     file = open(os.path.join(args.private_model_path, filename), "w+")
#     args.log_file = file
#     args.save_model_path = args.private_model_path
#     utils.augmented_print("##########################################", file)
#     utils.augmented_print(
#         "Training private models on '{}' dataset!".format(args.dataset), file
#     )
#     utils.augmented_print(
#         "Training private models on '{}' architecture!".format(args.architecture), file
#     )
#     utils.augmented_print(
#         "Number of private models: {:d}".format(args.num_models), file
#     )
#     utils.augmented_print(f"Initial learning rate: {args.lr}.", file)
#     utils.augmented_print(
#         "Number of epochs for training each model: {:d}".format(args.num_epochs), file
#     )
#
#     # Data loaders
#     if args.dataset_type == "imbalanced":
#         all_private_trainloaders = utils.load_private_data_imbalanced(args)
#     elif args.dataset_type == "balanced":
#         if args.balance_type == "standard":
#             all_private_trainloaders = utils.load_private_data(args=args)
#         elif args.balance_type == "perfect":
#             check_perfect_balance_type(args=args)
#             all_private_trainloaders = utils.load_private_data_imbalanced(args)
#         else:
#             raise Exception(f"Unknown args.balance_type: {args.balance_type}.")
#     else:
#         raise Exception(f"Unknown dataset type: {args.dataset_type}.")
#
#     evalloader = utils.load_evaluation_dataloader(args)
#     # evalloader = utils.load_private_data(args=args)[0]
#     print(f"eval dataset: ", evalloader.dataset)
#
#     if args.debug is True:
#         # Logs about the eval set
#         show_dataset_stats(
#             dataset=evalloader.dataset, args=args, file=file, dataset_name="eval"
#         )
#
#     # Training
#     summary = {
#         "loss": [],
#         "acc": [],
#         "balanced_acc": [],
#         "auc": [],
#     }
#     for id in range(args.begin_id, args.end_id):
#         utils.augmented_print("##########################################", file)
#
#         # Private model for initial training.
#         model = get_private_model_by_id(args=args, id=id)
#
#         if args.dataset == "pascal":
#             model_state_dict = model.state_dict()
#             pretrained_dict34 = torch.load("./architectures/resnet50-19c8e357.pth")
#             pretrained_dict_1 = {
#                 k: v for k, v in pretrained_dict34.items() if k in model_state_dict
#             }
#             model_state_dict.update(pretrained_dict_1)
#             model.load_state_dict(model_state_dict)
#
#         trainloader = all_private_trainloaders[id]
#
#         print(f"train dataset for model id: {id}", trainloader.dataset)
#
#         # Logs about the train set
#         if args.debug is True:
#             show_dataset_stats(
#                 dataset=trainloader.dataset,
#                 args=args,
#                 file=file,
#                 dataset_name="private train",
#             )
#         utils.augmented_print("Steps per epoch: {:d}".format(len(trainloader)), file)
#
#         if args.dataset.startswith("chexpert") and not args.architecture.startswith(
#             "densenet"
#         ):
#             devloader = get_chexpert_dev_loader(args=args)
#             result, best_model = train_chexpert.run(
#                 args=args,
#                 model=model,
#                 dataloader_train=trainloader,
#                 dataloader_dev=devloader,
#                 dataloader_eval=evalloader,
#             )
#         # elif args.dataset == 'cxpert':
#         #     train_cxpert(args=args, model=model, train_loader=trainloader,
#         #                  valid_loader=evalloader)
#         else:
#             train_model(
#                 args=args, model=model, trainloader=trainloader, evalloader=evalloader
#             )
#             result = eval_distributed_model(
#                 model=model, dataloader=evalloader, args=args
#             )
#
#         model_name = get_model_name_by_id(id=id)
#         result["model_name"] = model_name
#         result_str = from_result_to_str(result=result, sep=" | ", inner_sep=": ")
#         utils.augmented_print(text=result_str, file=file, flush=True)
#         summary = update_summary(summary=summary, result=result)
#
#         # Checkpoint
#         state = result
#         state["state_dict"] = model.state_dict()
#         filename = "checkpoint-{}.pth.tar".format(model_name)
#         filepath = os.path.join(args.private_model_path, filename)
#         torch.save(state, filepath)
#
#     utils.augmented_print("##########################################", file)
#
#     for key, value in summary.items():
#         if len(value) > 0:
#             avg_value = np.mean(value)
#             utils.augmented_print(f"Average {key} of private models: {avg_value}", file)
#
#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     utils.augmented_print(f"elapsed time: {elapsed_time}\n", file, flush=True)
#     utils.augmented_print("##########################################", file)
#     file.close()

##################
# NOISY ENSEMBLE #
##################
def evaluate_ensemble_model(args):
    """Evaluate the accuracy of noisy ensemble model under varying noise scales."""
    # Logs
    file = open(
        os.path.join(args.ensemble_model_path, "logs-ensemble(all).txt"), "w")
    utils.augmented_print("##########################################", file)
    utils.augmented_print(
        "Evaluating ensemble model 'ensemble(all)' on '{}' dataset!".format(
            args.dataset
        ),
        file,
    )
    utils.augmented_print(
        "Number of private models: {:d}".format(args.num_models), file
    )

    # Create an ensemble model
    private_models = load_private_models(args=args)
    ensemble_model = EnsembleModel(
        model_id=-1, args=args, private_models=private_models
    )
    # Evalloader
    evalloader = utils.load_evaluation_dataloader(args)
    # Different sigma values
    error_msg = (
        f"Unknown number of models: {args.num_models} for dataset {args.dataset}."
    )
    if args.dataset == "svhn":
        if args.num_models == 250:
            # sigma_list = [200, 150, 100, 50, 45, 40, 35, 30, 25, 20, 10, 5, 0]
            sigma_list = [args.sigma_gnmax]
        else:
            raise Exception(error_msg)
    elif args.dataset == "cifar10":
        if args.num_models == 50:
            # sigma_list = [40, 35, 30, 25, 20, 15, 14, 13, 12, 11, 10, 9, 8, 7,
            #               6, 5, 4, 3, 2, 1, 0]
            sigma_list = [args.sigma_gnmax]
        else:
            raise Exception(error_msg)
    elif args.dataset == "mnist":
        if args.num_models == 250:
            # sigma_list = [1, 0]
            # sigma_list = [
            #     200, 190, 180, 170, 160, 150, 140, 130, 120, 110, 100, 90, 80,
            #     70, 60, 50, 45, 40, 35, 30, 25, 20, 15, 14, 13, 12, 11, 10, 9,
            #     8, 7, 6, 5, 4, 3, 2, 1, 0]
            sigma_list = [args.sigma_gnmax]
        else:
            sigma_list = [x for x in range(100)]
            # raise Exception(error_msg)
    elif args.dataset == "fashion-mnist":
        # sigma_list = [50, 45, 40, 35, 30, 25, 20, 15, 14, 13, 12, 11, 10, 9,
        #               8, 7, 6, 5, 4, 3, 2, 1, 0]
        # sigma_list = [
        #     200, 190, 180, 170, 160, 150, 140, 130, 120, 110, 100, 90, 80,
        #     70, 60, 50, 45, 40, 35, 30, 25, 20, 15, 14, 13, 12, 11, 10, 9,
        #     8, 7, 6, 5, 4, 3, 2, 1, 0]
        sigma_list = [args.sigma_gnmax]
        # sigma_list = [x for x in range(10)]
    elif args.dataset == "cxpert":
        sigma_list = [args.sigma_gnmax]
    else:
        raise Exception(error_msg)

    accs = []
    gaps = []
    for sigma in sigma_list:
        args.sigma_gnmax = sigma
        acc, acc_detailed, gap, gap_detailed = ensemble_model.evaluate(
            evalloader, args)
        accs.append(acc)
        gaps.append(gap)
        utils.augmented_print("sigma_gnmax: {:.4f}".format(args.sigma_gnmax),
                              file)
        utils.augmented_print("Accuracy on evalset: {:.2f}%".format(acc), file)
        utils.augmented_print(
            "Detailed accuracy on evalset: {}".format(
                np.array2string(acc_detailed, precision=2, separator=", ")
            ),
            file,
        )
        utils.augmented_print(
            "Gap on evalset: {:.2f}% ({:.2f}|{:d})".format(
                100.0 * gap / args.num_models, gap, args.num_models
            ),
            file,
        )
        utils.augmented_print(
            "Detailed gap on evalset: {}".format(
                np.array2string(gap_detailed, precision=2, separator=", ")
            ),
            file,
            flush=True,
        )

    utils.augmented_print(f"Sigma list on evalset: {sigma_list}", file,
                          flush=True)
    utils.augmented_print(f"Accuracies on evalset: {accs}", file, flush=True)
    utils.augmented_print(f"Gaps on evalset: {gaps}", file, flush=True)
    utils.augmented_print("##########################################", file)
    file.close()

    if hasattr(private_models[0], "first_time"):
        model0 = private_models[0]
        print("first time: ", model0.first_time)
        print("middle time: ", model0.middle_time)
        print("last time: ", model0.last_time)


def evaluate_big_ensemble_model(args):
    """Query-answer process where each constituent model in the ensemble is
    big in the sense that we cannot load all the models to the GPUs at once."""
    # Logs
    file_name = "logs-evaluate-big-ensemble-(num-models:{})-(num-query-parties:{})-(query-mode:{})-(threshold:{:.1f})-(sigma-gnmax:{:.1f})-(sigma-threshold:{:.1f})-(budget:{:.2f}).txt".format(
        args.num_models,
        args.num_querying_parties,
        args.mode,
        args.threshold,
        args.sigma_gnmax,
        args.sigma_threshold,
        args.budget,
    )
    print("ensemble_model_path: ", args.ensemble_model_path)
    print("file_name: ", file_name)
    file = open(os.path.join(args.ensemble_model_path, file_name), "w")
    args.log_file = file
    # args.save_model_path = args.ensemble_model_path
    args.save_model_path = args.private_model_path
    utils.augmented_print("##########################################", file)
    utils.augmented_print(
        "Query-answer process on '{}' dataset!".format(args.dataset), file
    )
    utils.augmented_print(
        "Number of private models: {:d}".format(args.num_models), file
    )
    utils.augmented_print(
        "Number of querying parties: {:d}".format(args.num_querying_parties),
        file
    )
    utils.augmented_print("Querying mode: {}".format(args.mode), file)
    utils.augmented_print("Confidence threshold: {:.1f}".format(args.threshold),
                          file)
    utils.augmented_print(
        "Standard deviation of the Gaussian noise in the GNMax mechanism: {:.1f}".format(
            args.sigma_gnmax
        ),
        file,
    )
    utils.augmented_print(
        "Standard deviation of the Gaussian noise in the threshold mechanism: {:.1f}".format(
            args.sigma_threshold
        ),
        file,
    )
    utils.augmented_print(
        "Pre-defined privacy budget: ({:.2f}, {:.0e})-DP".format(
            args.budget, args.delta
        ),
        file,
    )
    utils.augmented_print("##########################################", file)

    all_models_id = -1
    big_ensemble = BigEnsembleModel(model_id=all_models_id, args=args)

    utils.augmented_print(
        "##########################################", file, flush=True
    )

    dataset_type = "test"
    if dataset_type == "dev":
        dataloader = utils.load_dev_dataloader(args=args)
    elif dataset_type == "test":
        dataloader = utils.load_evaluation_dataloader(
            args=args)  # gets full pascal test set from utils.py
        print("Loaded test set")
    else:
        raise Exception(f"Unsupported dataset_type: {dataset_type}.")
    print(f"dataset: ", dataloader.dataset)

    # Votes are returned from the individual models.
    # Voting based on the test set (not noisy).
    votes = big_ensemble.get_votes_cached(
        dataloader=dataloader, args=args, dataset_type=dataset_type
    )
    if args.class_type == 'multilabel_powerset':
        axis = 2
    else:
        axis = 1
    votes = pick_labels_general(labels=votes, args=args, axis=axis)

    # sigma_gnmax_list = [args.sigma_gnmax]
    # sigma_gnmax_list = [0]

    if args.command == 'evaluate_big_ensemble_model' and (
            args.class_type != 'multilabel_powerset'):
        sigma_gnmaxs = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
            18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
            33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 50, 55, 60,
            # 61, 62, 63, 64, 65, 66, 67, 68, 69, 70
        ]
    else:
        sigma_gnmaxs = [args.sigma_gnmax]
    thresholds = [args.threshold]
    sigma_thresholds = [args.sigma_threshold]
    # print('sigma_gnmax,balanced accuracy,number of answered queries')
    # sigma_gnmaxs = [
    #     1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    # sigma_gnmaxs = [
    #     21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
    #     33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 50, 55, 60]
    # sigma_gnmaxs = [0.625]:
    # sigma_gnmaxs = range(15, 20, 1)
    # sigma_gnmaxs = [5, 6, 7, 8, 9, 10]
    # for sigma_gnmax in sigma_gnmaxs:
    # header_printed = False
    # sigma_thresholds = [10, 15, 20, 25, 30, 35, 40]
    # sigma_thresholds = range(25, 51)
    # thresholds = [25, 30, 35, 40, 45]
    # thresholds = range(45, 56, 1)
    # sigma_thresholds = [0.01]
    # thresholds = [0.01]
    # sigma_gnmaxs = [23]
    is_header = False
    for sigma_gnmax in sigma_gnmaxs:
        for threshold in thresholds:
            pass
            # if threshold > args.num_models:
            #     # The threshold has to be lower than the number of labels.
            #     continue
            for sigma_threshold in sigma_thresholds:
                if sigma_threshold > threshold:
                    # The Gaussian noise sigma_threshold has to be lower than the threshold.
                    continue
                args.threshold = threshold
                args.sigma_threshold = sigma_threshold
                args.sigma_gnmax = sigma_gnmax

                indices_queried = np.arange(0, len(dataloader.dataset))
                results = big_ensemble.query(
                    queryloader=dataloader,
                    args=args,
                    indices_queried=indices_queried,
                    votes_queried=votes,
                )

                msg = {
                    "private_tau": args.private_tau,
                    "sigma-gnmax": sigma_gnmax,
                    "acc": results[metric.acc],
                    "balanced_accuracy": results[metric.balanced_acc],
                    "auc": results[metric.auc],
                    "map": results[metric.map],
                }
                msg_str = ";".join(
                    [f"{str(key)};{str(value)}" for key, value in msg.items()]
                )
                print(msg_str)

                num_labels = args.num_classes
                if args.pick_labels is not None and args.pick_labels != [-1]:
                    num_labels = len(args.pick_labels)

                file_name = (
                    f"evaluate_big_ensemble_{args.dataset}_{args.class_type}_"
                    f"summary_private_tau_{args.private_tau}_"
                    f"dataset_{args.dataset}_"
                    f"_private_tau_{args.private_tau}_"
                    f"labels_{num_labels}_"
                    f".txt"
                )
                with open(file_name, "a") as writer:
                    writer.write(msg_str + "\n")

                # file_name = (
                #     f"evaluate_big_ensemble_{args.class_type}_seaborn_"
                #     f"dataset_{args.dataset}_"
                #     f"_private_tau_{args.private_tau}_"
                #     f"labels_{num_labels}_"
                #     f"{args.timestamp}.txt"
                # )
                name = args.class_type
                dataset = args.dataset
                if dataset == 'celeba':
                    dataset = 'CelebA'
                file_name = f'labels_{name}_{dataset}_{num_labels}_labels.csv'
                if args.class_type != 'multilabel_powerset':
                    with open(file_name, "a") as writer:
                        if is_header is False:
                            is_header = True
                            writer.write('sigma,metric,value\n')

                        writer.write(
                            f"{args.sigma_gnmax},ACC,{results[metric.acc]}\n")
                        writer.write(
                            f"{args.sigma_gnmax},AUC,{results[metric.auc]}\n")
                        writer.write(
                            f"{args.sigma_gnmax},MAP,{results[metric.map]}\n")

                print(
                    "Note: we have the same balanced accuracy and auc because"
                    " we operate on votes and not the probability outputs."
                )
                results_str = utils.from_result_to_str(
                    result=utils.extract_metrics(results)
                )
                utils.augmented_print(results_str, file, flush=True)
                utils.print_metrics_detailed(results=results)

    file.close()


################
# QUERY-ANSWER #
################
def query_ensemble_model(args):
    """Query-answer process"""
    # Logs
    file_name = "logs-(num-models:{})-(num-query-parties:{})-(query-mode:{})-(threshold:{:.1f})-(sigma-gnmax:{:.1f})-(sigma-threshold:{:.1f})-(budget:{:.2f}).txt".format(
        args.num_models,
        args.num_querying_parties,
        args.mode,
        args.threshold,
        args.sigma_gnmax,
        args.sigma_threshold,
        args.budget,
    )
    print("ensemble_model_path: ", args.ensemble_model_path)
    print("file_name: ", file_name)
    file = open(os.path.join(args.ensemble_model_path, file_name), "w")
    args.save_model_path = args.ensemble_model_path
    utils.augmented_print("##########################################", file)
    utils.augmented_print(
        "Query-answer process on '{}' dataset!".format(args.dataset), file
    )
    utils.augmented_print(
        "Number of private models: {:d}".format(args.num_models), file
    )
    utils.augmented_print(
        "Number of querying parties: {:d}".format(args.num_querying_parties),
        file
    )
    utils.augmented_print("Querying mode: {}".format(args.mode), file)
    utils.augmented_print("Confidence threshold: {:.1f}".format(args.threshold),
                          file)
    utils.augmented_print(
        "Standard deviation of the Gaussian noise in the GNMax mechanism: {:.1f}".format(
            args.sigma_gnmax
        ),
        file,
    )
    utils.augmented_print(
        "Standard deviation of the Gaussian noise in the threshold mechanism: {:.1f}".format(
            args.sigma_threshold
        ),
        file,
    )
    utils.augmented_print(
        "Pre-defined privacy budget: ({:.2f}, {:.0e})-DP".format(
            args.budget, args.delta
        ),
        file,
    )
    utils.augmented_print("##########################################", file)

    model_path = args.private_model_path
    private_models = load_private_models(args=args, model_path=model_path)
    # Querying parties
    prev_num_models = args.num_models

    if args.test_virtual is True:
        query_ensemble_model_with_virtual_parties(args=args, file=file)

    parties_q = private_models[: args.num_querying_parties]
    args.querying_parties = parties_q

    # Answering parties.
    parties_a = []
    for i in range(args.num_querying_parties):
        # For a given querying party, skip this very querying party as its
        # own answering party.
        if args.test_virtual is True:
            num_private = len(private_models) // args.num_querying_parties
            start = i * num_private
            end = start + (i + 1) * num_private
            private_subset = private_models[0:start] + private_models[end:]
        else:
            private_subset = private_models[:i] + private_models[i + 1:]

        ensemble_model = EnsembleModel(
            model_id=i, private_models=private_subset, args=args
        )
        parties_a.append(ensemble_model)

    # Compute utility scores and sort available queries
    utils.augmented_print(
        "##########################################", file, flush=True
    )
    if args.attacker_dataset:
        unlabeled_dataset = utils.get_attacker_dataset(
            args=args, dataset_name=args.attacker_dataset
        )
        print("attacker uses {} dataset".format(args.attacker_dataset))
    else:
        unlabeled_dataset = utils.get_unlabeled_set(args=args)

    if args.mode == "random":
        all_indices = get_unlabeled_indices(args=args,
                                            dataset=unlabeled_dataset)
    else:
        unlabeled_dataloaders = utils.load_unlabeled_dataloaders(
            args=args, unlabeled_dataset=unlabeled_dataset
        )
        utility_scores = []

        # Select the utility function.
        if args.mode == "entropy":
            utility_function = compute_utility_scores_entropy
        elif args.mode == "gap":
            utility_function = compute_utility_scores_gap
        elif args.mode == "greedy":
            utility_function = compute_utility_scores_greedy
        elif args.mode == "deepfool":
            utility_function = compute_utility_scores_deepfool
        else:
            raise Exception(f"Unknown query selection mode: {args.mode}.")

        for i in range(args.num_querying_parties):
            filename = "{}-utility-scores-(mode-{})-dataset-{}.npy".format(
                parties_q[i].name, args.mode, args.dataset
            )
            filepath = os.path.join(args.ensemble_model_path, filename)
            if os.path.isfile(filepath) and args.debug is True:
                utils.augmented_print(
                    "Loading utility scores for '{}' in '{}' mode!".format(
                        parties_q[i].name, args.mode
                    ),
                    file,
                )
                utility = np.load(filepath)
            else:
                utils.augmented_print(
                    "Computing utility scores for '{}' in '{}' mode!".format(
                        parties_q[i].name, args.mode
                    ),
                    file,
                )
                utility = utility_function(
                    model=parties_q[i], dataloader=unlabeled_dataloaders[i],
                    args=args
                )
            utility_scores.append(utility)

        # Sort unlabeled data according to their utility scores.
        all_indices = []
        for i in range(args.num_querying_parties):
            offset = i * (
                    args.num_unlabeled_samples // args.num_querying_parties)
            indices = utility_scores[i].argsort()[::-1] + offset
            all_indices.append(indices)
            assert len(set(indices)) == len(indices)
        if not args.attacker_dataset:
            # this assertion seems only fails in entropy mode when using a different attacker dataset, is this okay?
            assert (
                    len(set(np.concatenate(all_indices, axis=0)))
                    == args.num_unlabeled_samples
            )

    utils.augmented_print(
        "##########################################", file, flush=True
    )
    utils.augmented_print(
        "Select queries according to their utility scores subject to the pre-defined privacy budget",
        file,
        flush=True,
    )

    for i in range(args.num_querying_parties):
        # Raw ensemble votes
        if args.attacker_dataset is None:
            attacker_dataset = ""
        else:
            attacker_dataset = args.attacker_dataset
        filename = "{}-raw-votes-(mode-{})-dataset-{}-attacker-{}.npy".format(
            parties_a[i].name, args.mode, args.dataset, attacker_dataset
        )
        filepath = os.path.join(args.ensemble_model_path, filename)
        utils.augmented_print(f"filepath: {filepath}", file=file)
        if os.path.isfile(filepath) and args.debug is True:
            utils.augmented_print(
                "Loading raw ensemble votes for '{}' in '{}' mode!".format(
                    parties_a[i].name, args.mode
                ),
                file,
            )
            votes = np.load(filepath)
        else:
            utils.augmented_print(
                "Generating raw ensemble votes for '{}' in '{}' mode!".format(
                    parties_a[i].name, args.mode
                ),
                file,
            )
            # Load unlabeled data according to a specific order
            unlabeled_dataloader_ordered = utils.load_ordered_unlabeled_data(
                args, all_indices[i], unlabeled_dataset=unlabeled_dataset
            )
            if args.vote_type == "confidence_scores":
                votes = parties_a[i].inference_confidence_scores(
                    unlabeled_dataloader_ordered, args
                )
            else:
                votes = parties_a[i].inference(unlabeled_dataloader_ordered,
                                               args)
            np.save(file=filepath, arr=votes)

        # Analyze how the pre-defined privacy budget will be exhausted when
        # answering queries.
        (
            max_num_query,
            dp_eps,
            partition,
            answered,
            order_opt,
        ) = analysis.analyze_privacy(votes=votes, args=args, file=file)

        utils.augmented_print("Querying party: {}".format(parties_q[i].name),
                              file)
        utils.augmented_print(
            "Maximum number of queries: {}".format(max_num_query), file
        )
        utils.augmented_print(
            "Privacy guarantee achieved: ({:.4f}, {:.0e})-DP".format(
                dp_eps[max_num_query - 1], args.delta
            ),
            file,
        )
        utils.augmented_print(
            "Expected number of queries answered: {:.3f}".format(
                answered[max_num_query - 1]
            ),
            file,
        )
        utils.augmented_print(
            "Partition of privacy cost: {}".format(
                np.array2string(
                    partition[max_num_query - 1], precision=3, separator=", "
                )
            ),
            file,
        )

        utils.augmented_print(
            "##########################################", file, flush=True
        )
        utils.augmented_print("Generate query-answer pairs.", file)
        indices_queried = all_indices[i][:max_num_query]
        queryloader = utils.load_ordered_unlabeled_data(
            args=args, indices=indices_queried,
            unlabeled_dataset=unlabeled_dataset
        )
        indices_answered, acc, acc_detailed, gap, gap_detailed = parties_a[
            i].query(
            queryloader, args, indices_queried
        )
        utils.save_raw_queries_targets(
            args=args,
            indices=indices_answered,
            dataset=unlabeled_dataset,
            name=parties_q[i].name,
        )
        utils.augmented_print("Accuracy on queries: {:.2f}%".format(acc), file)
        utils.augmented_print(
            "Detailed accuracy on queries: {}".format(
                np.array2string(acc_detailed, precision=2, separator=", ")
            ),
            file,
        )
        utils.augmented_print(
            "Gap on queries: {:.2f}% ({:.2f}|{:d})".format(
                100.0 * gap / len(parties_a[i].ensemble),
                gap,
                len(parties_a[i].ensemble),
            ),
            file,
        )
        utils.augmented_print(
            "Detailed gap on queries: {}".format(
                np.array2string(gap_detailed, precision=2, separator=", ")
            ),
            file,
        )

        utils.augmented_print(
            "##########################################", file, flush=True
        )
        utils.augmented_print("Check query-answer pairs.", file)
        queryloader = utils.load_ordered_unlabeled_data(
            args=args, indices=indices_answered,
            unlabeled_dataset=unlabeled_dataset
        )
        counts, ratios = utils.class_ratio(queryloader.dataset, args)
        utils.augmented_print(
            "Label counts: {}".format(np.array2string(counts, separator=", ")),
            file
        )
        utils.augmented_print(
            "Class ratios: {}".format(
                np.array2string(ratios, precision=2, separator=", ")
            ),
            file,
        )
        utils.augmented_print(
            "Number of samples: {:d}".format(len(queryloader.dataset)), file
        )
        utils.augmented_print(
            "##########################################", file, flush=True
        )
    file.close()
    args.num_models = prev_num_models


def query_big_ensemble_model(args):
    """Query-answer process where each constituent model in the ensemble is
    big in the sense that we cannot load all the models to the GPUs at once."""
    # Logs
    file_name = "logs-(num-models:{})-(num-query-parties:{})-(query-mode:{})-(threshold:{:.1f})-(sigma-gnmax:{:.1f})-(sigma-threshold:{:.1f})-(budget:{:.2f}).txt".format(
        args.num_models,
        args.num_querying_parties,
        args.mode,
        args.threshold,
        args.sigma_gnmax,
        args.sigma_threshold,
        args.budget,
    )
    print("ensemble_model_path: ", args.ensemble_model_path)
    print("file_name: ", file_name)
    log_file = open(os.path.join(args.ensemble_model_path, file_name), "w")
    args.log_file = log_file
    # args.save_model_path = args.ensemble_model_path
    args.save_model_path = args.private_model_path
    utils.augmented_print("##########################################",
                          log_file)
    utils.augmented_print(
        "Query-answer process on '{}' dataset!".format(args.dataset), log_file
    )
    utils.augmented_print(
        "Number of private models: {:d}".format(args.num_models), log_file
    )
    utils.augmented_print(
        "Number of querying parties: {:d}".format(args.num_querying_parties),
        log_file
    )
    utils.augmented_print("Querying mode: {}".format(args.mode), log_file)
    utils.augmented_print(
        "Confidence threshold: {:.1f}".format(args.threshold), log_file
    )
    utils.augmented_print(
        "Standard deviation of the Gaussian noise in the GNMax mechanism: {:.1f}".format(
            args.sigma_gnmax
        ),
        log_file,
    )
    utils.augmented_print(
        "Standard deviation of the Gaussian noise in the threshold mechanism: {:.1f}".format(
            args.sigma_threshold
        ),
        log_file,
    )
    utils.augmented_print(
        "Pre-defined privacy budget: ({:.2f}, {:.0e})-DP".format(
            args.budget, args.delta
        ),
        log_file,
    )
    utils.augmented_print("##########################################",
                          log_file)

    # Answering parties
    parties_a = {}
    if args.num_querying_parties > 0:
        for i in range(args.num_querying_parties):
            # For a given querying party, skip this very querying party as its
            # own answering party.
            ensemble_model = BigEnsembleModel(model_id=i, args=args)
            parties_a[i] = ensemble_model
            args.querying_parties = range(args.num_querying_parties)
    else:
        # Special case when we have to train on all models from a given dataset.
        # This is for the medical datasets with different training models and
        # datasets.
        other_querying_party = -1
        assert args.num_querying_parties == other_querying_party
        ensemble_model = BigEnsembleModel(model_id=other_querying_party,
                                          args=args)
        querying_party_ids = args.querying_party_ids
        for querying_party_id in querying_party_ids:
            parties_a[querying_party_id] = ensemble_model
        args.querying_parties = querying_party_ids

    utils.augmented_print(
        "##########################################", log_file, flush=True
    )
    utils.augmented_print(
        "Compute utility scores and sort available queries.", file=log_file
    )
    # Utility functions
    if args.mode == "entropy":
        utility_function = compute_utility_scores_entropy
    elif args.mode == "gap":
        utility_function = compute_utility_scores_gap
    elif args.mode == "greedy":
        utility_function = compute_utility_scores_greedy
    elif args.mode == "deepfool":
        utility_function = compute_utility_scores_deepfool
    else:
        assert args.mode == "random"
        utility_function = None

    unlabeled_dataset = get_unlabeled_set(args=args)

    if args.mode != "random":
        # Dataloaders
        unlabeled_dataloaders = utils.load_unlabeled_dataloaders(args=args)
        # Utility scores
        utility_scores = []
        for i in range(args.num_querying_parties):
            query_party_name = get_model_name_by_id(id=i)
            filename = "{}-utility-scores-(mode:{}).npy".format(
                query_party_name, args.mode
            )
            if os.name == "nt":
                filename = "{}-utility-scores-(mode_{}).npy".format(
                    query_party_name, args.mode
                )
            filepath = os.path.join(args.ensemble_model_path, filename)
            if os.path.isfile(filepath):
                utils.augmented_print(
                    "Loading utility scores for '{}' in '{}' mode!".format(
                        query_party_name, args.mode
                    ),
                    log_file,
                )
                utility = np.load(filepath)
            else:
                utils.augmented_print(
                    "Computing utility scores for '{}' in '{}' mode!".format(
                        query_party_name, args.mode
                    ),
                    log_file,
                )
                query_party_model = load_private_model_by_id(args=args, id=i)
                utility = utility_function(
                    model=query_party_model,
                    dataloader=unlabeled_dataloaders[i],
                    args=args,
                )
            utility_scores.append(utility)
        # Sort unlabeled data according to their utility scores.
        unlabeled_indices = []
        for i in range(args.num_querying_parties):
            offset = i * (
                    args.num_unlabeled_samples // args.num_querying_parties)
            indices = utility_scores[i].argsort()[::-1] + offset
            unlabeled_indices.append(indices)
            assert len(set(indices)) == len(indices)
        if not args.attacker_dataset:
            # This assertion seems only fails in entropy mode when using a
            # different attacker dataset, is this okay? TODO
            assert (
                    len(set(np.concatenate(unlabeled_indices, axis=0)))
                    == args.num_unlabeled_samples
            )
    else:
        # Select the queries randomly.
        unlabeled_indices = get_unlabeled_indices(args=args,
                                                  dataset=unlabeled_dataset)

    utils.augmented_print(
        "##########################################", log_file, flush=True
    )
    utils.augmented_print(
        "Select queries according to their utility scores subject to the "
        "pre-defined privacy budget.",
        log_file,
        flush=True,
    )
    utils.augmented_print(
        "Analyze how the pre-defined privacy budget will be exhausted when "
        "answering queries.",
        log_file,
        flush=True,
    )

    if args.class_type == "multiclass":
        if args.threshold == 0:
            assert args.sigma_threshold == 0
            analyze = analysis.analyze_multiclass_gnmax
        else:
            analyze = analysis.analyze_multiclass_confident_gnmax
    elif args.class_type == "multiclass_confidence":
        analyze = analysis.analyze_multiclass_confident_gnmax_confidence_scores
    elif args.class_type == "multilabel":
        analyze = analysis.analyze_multilabel
    elif args.class_type == "multilabel_counting":
        analyze = analysis.analyze_multilabel_counting
    elif args.class_type == "multilabel_counting_gaussian":
        analyze = analysis.analyze_multilabel_counting
    elif args.class_type == "multilabel_counting_laplace":
        analyze = analysis.analyze_multilabel_counting
    elif args.class_type == "multilabel_tau":
        # The multilabel tau from the Priate kNN.
        analyze = analysis.analyze_multilabel_tau
    elif args.class_type == "multilabel_tau_data_independent":
        # Use PATE RDP for the data-independent analysis_test of multilabel
        # classification.
        analyze = analysis.analyze_multilabel_tau_data_independent
    elif args.class_type == "multilabel_tau_dep":
        analyze = analysis.analyze_multilabel
    elif args.class_type == "multilabel_pate":
        analyze = analysis.analyze_multilabel_pate
    elif args.class_type == "multilabel_tau_pate":
        analyze = analysis.analyze_tau_pate
    elif args.class_type == "multilabel_powerset":
        analyze = analysis.analyze_multilabel_powerset
    else:
        raise Exception(f"Unknown args.class_type: {args.class_type}.")

    for party_nr, party_id in enumerate(args.querying_parties):
        big_ensemble = parties_a[party_id]
        party_unlabeled_indices = unlabeled_indices[party_nr]
        query_party_name = get_model_name_by_id(id=party_id)
        utils.augmented_print(f"Querying party: {query_party_name}", log_file)

        # Load unlabeled data according to a specific order.
        unlabeled_dataloader_ordered = utils.load_ordered_unlabeled_data(
            args, party_unlabeled_indices, unlabeled_dataset=unlabeled_dataset
        )

        dataset_type = "unlabeled"
        # dataset_type = "test"
        all_votes = big_ensemble.get_votes_cached(
            dataloader=unlabeled_dataloader_ordered,
            args=args,
            dataset_type=dataset_type,
            party_id=party_id,
        )
        if args.class_type == 'multilabel_powerset':
            axis = 2
        else:
            axis = 1
        votes = pick_labels_general(labels=all_votes, args=args, axis=axis)

        num_samples = len(all_votes)
        num_teachers = args.num_models
        num_labels = len(args.pick_labels)
        # num_classes = 2 ** num_labels
        case = None
        # case = 'all_negative_votes'
        # case = 'random_votes'
        # case = 'intermediate'
        if case is not None:
            if case == 'all_negative_votes':
                # All same votes - all are negative.
                votes = np.zeros(
                    (num_samples, num_teachers, num_labels)).astype(
                    int)
            elif case == 'random_votes':
                # Random votes (all votes are generated randomly).
                votes = torch.randint(low=0, high=2, size=(
                    num_samples, num_teachers, num_labels)).numpy()
            elif case == 'intermediate':
                # How many labels are random out of all possible labels?
                random_rate = 0.0
                num_labels_random = int(num_labels * random_rate)
                num_labels_all_negative = num_labels - num_labels_random
                votes_all_negative = np.zeros(
                    (num_samples, num_teachers, num_labels_all_negative)
                ).astype(int)
                votes_random = torch.randint(low=0, high=2, size=(
                    num_samples, num_teachers, num_labels_random)).numpy()
                votes = np.concatenate([votes_all_negative, votes_random],
                                       axis=-1)
            else:
                raise Exception(f"Unsupported case: {case}.")
            if args.class_type == 'multilabel':
                votes_sum = votes.sum(axis=1)
                positive_votes = votes_sum
                negative_votes = 50 - votes_sum
                votes = np.concatenate(
                    [negative_votes[:, :, np.newaxis],
                     positive_votes[:, :, np.newaxis]],
                    axis=-1)
            elif args.class_type == 'multilabel_powerset':
                pass
            else:
                raise Exception(
                    f"Unsupported args.class_type: {args.class_type}.")

        if args.debug:
            pass
        # ensemble_vote_limit = 500
        # utils.augmented_print(
        #     text=f"initial vote shape: {votes.shape}", file=log_file)
        # utils.augmented_print(
        #     text=f"debug - ensemble vote limit: {ensemble_vote_limit}",
        #     file=log_file)
        # votes = votes[:ensemble_vote_limit]
        # if args.class_type in ['multilabel', 'multilabel_counting']:
        #     pass
        #     targets = utils.get_all_targets_numpy(
        #         dataloader=unlabeled_dataloader_ordered, args=args)
        #     start = 1
        #     # start = ensemble_vote_limit
        #     for limit in range(start, ensemble_vote_limit + 1):
        #         current_targets = targets[:limit]
        #         current_votes = votes[:limit]
        #         results = big_ensemble.get_multilabel_balanced_acc_from_votes(
        #             votes=current_votes, targets=current_targets, args=args)
        #         balanced_acc, balanced_acc_detailed = results
        #         print(limit, ',', balanced_acc)
        #         # print('balanced acc detailed: ', balanced_acc_detailed)
        utils.augmented_print(
            text=f"shape of votes: {votes.shape}", file=log_file, flush=True
        )

        # header_printed = False

        sigma_gnmaxs = [args.sigma_gnmax]
        thresholds = [args.threshold]
        sigma_thresholds = [args.sigma_threshold]

        # sigma_gnmaxs = np.repeat(args.sigma_gnmax, 100)
        # sigma_gnmaxs = np.array([x for x in range(1, 21, 1)])
        # sigma_gnmaxs = np.linspace(7, 70, 1000)
        # privacy_budgets = [args.budget]
        # privacy_budgets = np.array([x for x in range(1, 1000, 1)])
        # print('sigma_gnmax,balanced accuracy,number of answered queries')
        # sigma_gnmaxs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 20,]
        # sigma_gnmaxs = [0.625]:
        # sigma_gnmaxs = range(15, 20, 1)
        # sigma_gnmaxs = [5, 6, 7, 8, 9, 10]
        # for sigma_gnmax in sigma_gnmaxs:
        # sigma_gnmaxs = [x for x in range(1, 26)]
        # sigma_thresholds = [10, 15, 20, 25, 30, 35, 40]
        # sigma_thresholds = range(25, 56)
        # thresholds = [25, 30, 35, 40, 45]
        # thresholds = range(30, 56, 1)
        # sigma_thresholds = [0.01]
        # thresholds = [0.01]
        # sigma_gnmaxs = [7]
        # private_query_counts = [args.private_query_count]
        # private_query_counts = np.array([x for x in range(0, 1001, 1)])
        # sigma_gnmaxs = [
        #     1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
        #     18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
        #     33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 50, 55, 60]
        # sigma_gnmaxs = [x for x in range(1, 30)]
        # sigma_gnmaxs = [x for x in range(1, )]
        for sigma_gnmax in sigma_gnmaxs:
            # for private_query_count in private_query_counts:
            #     args.private_query_count = private_query_count
            # for privacy_budget in privacy_budgets:
            #     args.budget = privacy_budget
            for threshold in thresholds:
                pass
                # if threshold > args.num_models:
                #     # The threshold has to be lower than the number of labels.
                #     continue
                for sigma_threshold in sigma_thresholds:
                    if sigma_threshold > threshold:
                        # The Gaussian noise sigma_threshold has to be lower
                        # than the threshold.
                        continue
                    args.threshold = threshold
                    args.sigma_threshold = sigma_threshold
                    args.sigma_gnmax = sigma_gnmax
                    # TODO remove votes for unsupported labels
                    # Determine what is the max number of queries to answer.
                    max_num_query, dp_eps, partition, answered, order_opt = analyze(
                        votes=votes,
                        threshold=args.threshold,
                        sigma_threshold=args.sigma_threshold,
                        sigma_gnmax=args.sigma_gnmax,
                        args=args,
                        budget=args.budget,
                        delta=args.delta,
                        file=log_file,
                    )
                    print(sigma_gnmax, ',', max_num_query)
                    # continue
                    if max_num_query == 0:
                        # continue
                        raise Exception(
                            f"No queries answered. The privacy "
                            f"budget is too low: {args.budget}.")
                    # assert max_num_query > 0, "Check the sigma_gnmax, it might be too small."
                    # utils.augmented_print(
                    #     "Maximum number of queries: {}".format(max_num_query), log_file)
                    # utils.augmented_print(
                    #     "Privacy guarantee achieved: ({:.4f}, {:.0e})-DP".format(
                    #         dp_eps[max_num_query - 1], args.delta), log_file)
                    # utils.augmented_print(
                    #     "Expected number of queries answered: {:.3f}".format(
                    #         answered[max_num_query - 1]), log_file)
                    # utils.augmented_print("Partition of privacy cost: {}".format(
                    #     np.array2string(partition[max_num_query - 1], precision=3,
                    #                     separator=', ')), log_file)
                    # utils.augmented_print("##########################################",
                    #                       log_file, flush=True)
                    # utils.augmented_print("Generate query-answer pairs.", log_file)

                    indices_queried = party_unlabeled_indices[:max_num_query]

                    if args.class_type in ["multiclass_confidence",
                                           "multilabel_powerset"]:
                        # The votes are softmax values for each teacher and
                        # data point: votes are of shape
                        # (num_teachers, num_data_points, num_classes).
                        votes_queried = votes[:, :max_num_query, :]
                    else:
                        votes_queried = votes[:max_num_query]

                    # if args.debug is True:
                    #     if args.class_type in ['multilabel', 'multilabel_counting']:
                    #         pass
                    #         targets_queried = targets[:max_num_query]
                    #         results = big_ensemble.get_multilabel_balanced_acc_from_votes(
                    #             votes=votes_queried, targets=targets_queried, args=args)
                    #         balanced_acc, balanced_acc_detailed = results
                    #         print('balanced acc: ', balanced_acc)
                    #         print('balanced acc detailed: ', balanced_acc_detailed)
                    queryloader = utils.load_ordered_unlabeled_data(
                        args=args,
                        indices=indices_queried,
                        unlabeled_dataset=unlabeled_dataset,
                    )

                    results = big_ensemble.query(
                        queryloader=queryloader,
                        args=args,
                        indices_queried=indices_queried,
                        votes_queried=votes_queried,
                    )

                    # Get the incurred privacy budget.
                    if isinstance(dp_eps, np.ndarray):
                        if max_num_query > 0:
                            dp_eps = dp_eps[max_num_query - 1]
                        else:
                            dp_eps = 0

                    msg = {
                        "private_tau": args.private_tau,
                        "privacy_budget": args.budget,
                        "max_num_query": max_num_query,
                        "dp_eps": dp_eps,
                        "sigma-gnmax": sigma_gnmax,
                        "acc": results[metric.acc],
                        "balanced_accuracy": results[metric.balanced_acc],
                        "auc": results[metric.auc],
                        "map": results[metric.map],
                        "num_answered_queries": len(
                            results[result.indices_answered]),
                        "num_labels_answered": results[result.count_answered],
                    }
                    msg_str = ";".join(
                        [f"{str(key)};{str(value)}" for key, value in
                         msg.items()]
                    )
                    print(msg_str)
                    with open(
                            "query_big_ensemble_summary_private_tau_all.txt",
                            "a"
                    ) as writer:
                        writer.write(msg_str + "\n")
                    with open(
                            f"query_big_ensemble_{args.dataset}_"
                            f"summary_private_tau_{args.private_tau}_"
                            f"{args.class_type}.txt",
                            "a",
                    ) as writer:
                        writer.write(
                            f"{args.private_tau},ACC,{results[metric.acc]}\n")
                        writer.write(
                            f"{args.private_tau},AUC,{results[metric.auc]}\n")
                        writer.write(
                            f"{args.private_tau},MAP,{results[metric.map]}\n")

                    with open(
                            f"query_big_ensemble_{args.dataset}_"
                            f"{args.private_tau}_answered_epsilon_method.txt",
                            "a",
                    ) as writer:
                        if args.class_type == "multilabel":
                            method = "PATE"
                        elif args.class_type in [
                            "multilabel_tau",
                            "multilabel_tau_pate",
                        ]:
                            method = f"L{args.private_tau_norm}"
                        else:
                            method = args.class_type

                        writer.write(f"{max_num_query},{dp_eps},{method}\n")
                    aggregated_labels = results[result.predictions]
                    indices_answered = results[result.indices_answered]
                    aggregated_labels = aggregated_labels[indices_answered]
                    # balanced_acc = results.get(metric.balanced_acc, None)
                    # count_answered = results.get(result.count_answered, None)

                    #
                    # if count_answered is None:
                    #     count_answered = 'N/A'

                    # header = ['epsilon',
                    #           'max_num_query',
                    #           'sigma_gnmax',
                    #           'threshold',
                    #           'sigma_threshold',
                    #           'balanced_acc',
                    #           'num_labels_answered', 'budget']
                    # data = [dp_eps, max_num_query, args.sigma_gnmax,
                    #         args.threshold, args.sigma_threshold,
                    #         balanced_acc, count_answered, args.budget]

                    # print(sigma_gnmax, ',', balanced_acc, ',', max_num_query,
                    #       ',', num_labels_answered)

                    # if not header_printed:
                    #     print(args.sep.join(header))
                    #     header_printed = True
                    # print(args.sep.join([str(x) for x in data]))
        # print("AGG labels", aggregated_labels)
        # print("AGG labels size", aggregated_labels.shape)
        # print("indices answered", len(indices_answered))
        utils.save_labels(name=query_party_name, args=args,
                          labels=aggregated_labels)
        if args.query_set_type == "raw":
            utils.save_raw_queries_targets(
                args=args,
                indices=indices_answered,
                dataset=unlabeled_dataset,
                name=query_party_name,
            )
        elif args.query_set_type == "numpy":
            utils.save_queries(
                args=args,
                indices=indices_answered,
                dataset=unlabeled_dataset,
                name=query_party_name,
            )
        else:
            raise Exception(
                f"Unknown type of the query dataset for retraining: "
                f"{args.query_set_type}."
            )

        utils.augmented_print(
            "##########################################", log_file, flush=True
        )
        utils.augmented_print("Check query-answer pairs.", log_file)

        utils.augmented_print(
            utils.from_result_to_str(result=utils.extract_metrics(results)),
            log_file,
            flush=True,
        )

        if args.debug is True:
            queryloader = utils.load_ordered_unlabeled_data(
                args=args, indices=indices_answered,
                unlabeled_dataset=unlabeled_dataset
            )
            counts, ratios = utils.class_ratio(queryloader.dataset, args)
            utils.augmented_print(
                "Label counts: {}".format(
                    np.array2string(counts, separator=", ")),
                log_file,
            )
            utils.augmented_print(
                "Class ratios: {}".format(
                    np.array2string(ratios, precision=2, separator=", ")
                ),
                log_file,
            )
            utils.augmented_print(
                "Number of samples: {:d}".format(len(queryloader.dataset)),
                log_file
            )
        utils.augmented_print(
            "##########################################", log_file, flush=True
        )

    log_file.close()


############################
# RETRAIN PRIVATE MODELS   #
############################
def retrain_private_models(args):
    """
    Retrain N = num-querying-parties private models.

    :arg args: program parameters
    """
    assert 0 <= args.begin_id and args.begin_id < args.end_id and args.end_id

    if args.num_querying_parties > 0:
        args.querying_parties = range(args.begin_id, args.end_id, 1)
    else:
        other_querying_party = -1
        assert args.num_querying_parties == other_querying_party
        args.querying_parties = args.querying_party_ids

    # Logs
    filename = 'logs-(num_models:{:d})-(id:{:d}-{:d})-(num-epochs:{:d})-(budget:{:f})-(dataset:{})-(architecture:{}).txt'.format(
        args.num_models,
        args.begin_id + 1, args.end_id,
        args.num_epochs,
        args.budget,
        args.dataset,
        args.architecture,
    )
    print('filename: ', filename)
    file = open(os.path.join(args.retrained_private_model_path, filename), 'w')
    args.save_model_path = args.retrained_private_model_path
    utils.augmented_print("##########################################", file)
    utils.augmented_print(
        "Retraining the private models of all querying parties on '{}' dataset!".format(
            args.dataset), file)
    utils.augmented_print(
        "Number of querying parties: {:d}".format(len(args.querying_parties)),
        file)
    utils.augmented_print("Initial learning rate: {:.2f}".format(args.lr), file)
    utils.augmented_print(
        "Number of epochs for retraining each model: {:d}".format(
            args.num_epochs), file)
    if args.test_virtual:
        assert args.num_querying_parties > 0
        prev_num_models = args.num_models
        args.num_models = args.num_querying_parties
        if args.dataset_type == 'imbalanced':
            all_private_trainloaders = utils.load_private_data_imbalanced(args)
        elif args.dataset_type == 'balanced':
            all_private_trainloaders = utils.load_private_data(args)
        else:
            raise Exception(
                'Unknown dataset type: {}'.format(args.dataset_type))
        evalloader = utils.load_evaluation_dataloader(args)
    # Dataloaders
    if args.dataset_type == 'imbalanced':
        all_augmented_dataloaders = utils.load_private_data_and_qap_imbalanced(
            args=args)
    elif args.dataset_type == 'balanced':
        if args.balance_type == 'standard':
            all_augmented_dataloaders = utils.load_private_data_and_qap(
                args=args)
        elif args.balance_type == 'perfect':
            check_perfect_balance_type(args=args)
            all_augmented_dataloaders = utils.load_private_data_and_qap_imbalanced(
                args=args)
        else:
            raise Exception(f'Unknown args.balance_type: {args.balance_type}.')
    else:
        raise Exception(f'Unknown dataset type: {args.dataset_type}.')
    evalloader = utils.load_evaluation_dataloader(args)
    # Training
    for party_nr, party_id in enumerate(args.querying_parties):
        utils.augmented_print("##########################################",
                              file)
        # Different random seeds.
        # seed_list = [11, 13, 17, 113, 117]
        # seed_list = [11, 13, 17]
        seed_list = [args.seed]
        model_name = get_model_name_by_id(id=party_id)
        summary = {
            metric.loss: [],
            metric.acc: [],
            metric.balanced_acc: [],
            metric.auc: [],
            metric.acc_detailed: [],
            metric.balanced_acc_detailed: [],
        }

        trainloader = all_augmented_dataloaders[party_nr]
        show_dataset_stats(
            dataset=trainloader.dataset,
            args=args,
            dataset_name='retrain data',
            file=file)

        model = None
        for seed in seed_list:
            args.seed = seed
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            if args.cuda:
                torch.cuda.manual_seed(args.seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

            if args.retrain_model_type == 'load':
                # Here. Need pretrained imagenet
                model = load_private_model_by_id(
                    args=args, id=party_id, model_path=args.private_model_path)
            elif args.retrain_model_type == 'raw':
                model = get_private_model_by_id(args=args, id=party_id)
                model.name = model_name
            else:
                raise Exception(f"Unknown args.retrain_model_type: "
                                f"{args.retrain_model_type}")

            # Load the pre-trained models for re-training
            if args.dataset == 'pascal':
                model_name = f'mutillabel_net_params_{party_id}.pkl'
                model_path = args.private_model_path
                filepath = os.path.join(model_path, model_name)

                checkpoint = torch.load(filepath)
                model.load_state_dict(checkpoint)

            # Private model for re-training.
            train_model(args=args, model=model, trainloader=trainloader,
                        evalloader=evalloader)

            result = eval_model(model=model, dataloader=evalloader, args=args)
            summary = update_summary(summary=summary, result=result)

        # Add more info about the parameters.
        summary['model_name'] = model_name
        from_args = ['dataset', 'num_models', 'budget', 'architecture']
        for arg in from_args:
            summary[arg] = getattr(args, arg)

        # Aggregate results from different seeds.
        for metric_key in [metric.loss, metric.acc, metric.balanced_acc,
                           metric.auc]:
            value = summary[metric_key]
            if len(value) > 0:
                avg_value = np.mean(value)
                summary[metric_key] = avg_value
            else:
                summary[metric_key] = 'N/A'

        for metric_key in [metric.acc_detailed, metric.balanced_acc_detailed]:
            detailed_value = summary[metric_key]
            if len(detailed_value) > 0:
                detailed_value = np.array(detailed_value)
                summary[metric_key] = detailed_value.mean(axis=0)
                summary[metric_key.name + '_std'] = detailed_value.std(axis=0)
            else:
                summary[metric_key] = 'N/A'

        summary_str = from_result_to_str(result=summary, sep=' | ',
                                         inner_sep=': ')
        utils.augmented_print(text=summary_str, file=file, flush=True)

        if model is not None:
            utils.save_model(args=args, model=model, result_test=summary)

        utils.augmented_print("##########################################",
                              file)

    utils.augmented_print("##########################################", file)

    file.close()

    if args.test_virtual:
        args.num_models = prev_num_models


def train_student_model(args):
    """
    Retrain N = num-querying-parties private models.

    :arg args: program parameters
    """
    assert 0 <= args.begin_id and args.begin_id < args.end_id and args.end_id

    if args.num_querying_parties > 0:
        args.querying_parties = range(args.begin_id, args.end_id, 1)
    else:
        other_querying_party = -1
        assert args.num_querying_parties == other_querying_party
        args.querying_parties = args.querying_party_ids

    # Logs
    filename = "logs-(num_models:{:d})-(id:{:d}-{:d})-(num-epochs:{:d})-(budget:{:f})-(dataset:{})-(architecture:{}).txt".format(
        args.num_models,
        args.begin_id + 1,
        args.end_id,
        args.num_epochs,
        args.budget,
        args.dataset,
        args.architecture,
    )
    print("filename: ", filename)
    file = open(os.path.join(args.retrained_private_model_path, filename), "w")
    args.save_model_path = args.retrained_private_model_path
    utils.augmented_print("##########################################", file)
    utils.augmented_print(
        "Retraining the private models of all querying parties on '{}' dataset!".format(
            args.dataset
        ),
        file,
    )
    utils.augmented_print(
        "Number of querying parties: {:d}".format(len(args.querying_parties)),
        file
    )
    utils.augmented_print("Initial learning rate: {:.2f}".format(args.lr), file)
    utils.augmented_print(
        "Number of epochs for retraining each model: {:d}".format(
            args.num_epochs), file
    )
    if args.test_virtual:
        assert args.num_querying_parties > 0
        prev_num_models = args.num_models
        args.num_models = args.num_querying_parties
        if args.dataset_type == "imbalanced":
            all_private_trainloaders = utils.load_private_data_imbalanced(args)
        elif args.dataset_type == "balanced":
            all_private_trainloaders = utils.load_private_data(args)
        else:
            raise Exception(
                "Unknown dataset type: {}".format(args.dataset_type))
        evalloader = utils.load_evaluation_dataloader(args)
    # Dataloaders
    if args.dataset_type == "imbalanced":
        all_augmented_dataloaders = utils.load_private_data_and_qap_imbalanced(
            args=args
        )
    elif args.dataset_type == "balanced":
        if args.balance_type == "standard":
            all_augmented_dataloaders = utils.load_private_data_and_qap(
                args=args)
        elif args.balance_type == "perfect":
            check_perfect_balance_type(args=args)
            all_augmented_dataloaders = utils.load_private_data_and_qap_imbalanced(
                args=args
            )
        else:
            raise Exception(f"Unknown args.balance_type: {args.balance_type}.")
    else:
        raise Exception(f"Unknown dataset type: {args.dataset_type}.")
    evalloader = utils.load_evaluation_dataloader(args)
    # Training
    utils.augmented_print("##########################################", file)
    # Different random seeds.
    # seed_list = [11, 13, 17, 113, 117]
    # seed_list = [11, 13, 17]
    seed_list = [args.seed]
    model_name = get_model_name_by_id(id=0)
    summary = {
        metric.loss: [],
        metric.acc: [],
        metric.balanced_acc: [],
        metric.auc: [],
        metric.map: [],
        metric.acc_detailed: [],
        metric.balanced_acc_detailed: [],
        metric.auc_detailed: [],
        metric.map_detailed: []
    }
    trainloader = all_augmented_dataloaders[0]
    # print("len trainloader", len(trainloader))
    # print("attr", trainloader.dataset.__dict__.keys())
    # show_dataset_stats(
    #     dataset=trainloader.dataset, args=args, dataset_name="retrain data", file=file
    # )
    if args.dataset == "pascal" and args.retrain_fine_tune:
        model = resnetpre()
        print("Loaded pretrained resnet50")
    elif args.dataset == "cxpert" and args.retrain_fine_tune:
        model = densenetpre()
        print("Loaded pretrained densenet")
    else:
        if args.retrain_model_type == 'load':
            model = load_private_model_by_id(
                args=args, id=0, model_path=args.private_model_path)
        elif args.retrain_model_type == 'raw':
            model = get_private_model_by_id(args=args, id=0)
            model.name = model_name
        else:
            raise Exception(f"Unknown args.retrain_model_type: "
                            f"{args.retrain_model_type}")

    args.seed = seed_list[0]
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    train_model(args=args, model=model, trainloader=trainloader,
                evalloader=evalloader)

    result = eval_model(model=model, dataloader=evalloader, args=args)
    summary = update_summary(summary=summary, result=result)
    summary["model_name"] = model_name
    from_args = ["dataset", "num_models", "budget", "architecture"]
    for arg in from_args:
        summary[arg] = getattr(args, arg)

    # Aggregate results from different seeds.
    for metric_key in [metric.loss, metric.acc, metric.balanced_acc, metric.auc,
                       metric.map]:
        value = summary[metric_key]
        if len(value) > 0:
            avg_value = np.mean(value)
            summary[metric_key] = avg_value
        else:
            summary[metric_key] = "N/A"

    for metric_key in [metric.acc_detailed, metric.balanced_acc_detailed,
                       metric.auc_detailed, metric.map_detailed]:
        detailed_value = summary[metric_key]
        if len(detailed_value) > 0:
            detailed_value = np.array(detailed_value)
            summary[metric_key] = detailed_value.mean(axis=0)
            summary[metric_key.name + "_std"] = detailed_value.std(axis=0)
        else:
            summary[metric_key] = "N/A"

    summary_str = from_result_to_str(result=summary, sep=" | ", inner_sep=": ")
    utils.augmented_print(text=summary_str, file=file, flush=True)

    if model is not None:
        utils.save_model(args=args, model=model, result_test=summary)

    utils.augmented_print("##########################################", file)

    utils.augmented_print("##########################################", file)

    file.close()


def test_models(args):
    start_time = time.time()

    if args.num_querying_parties > 0:
        # Checks
        assert 0 <= args.begin_id
        assert args.begin_id < args.end_id
        assert args.end_id <= args.num_models
        args.querying_parties = range(args.begin_id, args.end_id, 1)
    else:
        other_querying_party = -1
        assert args.num_querying_parties == other_querying_party
        args.querying_parties = args.querying_party_ids

    # Logs
    filename = "logs-testing-(id:{:d}-{:d})-(num-epochs:{:d}).txt".format(
        args.begin_id + 1, args.end_id, args.num_epochs
    )
    file = open(os.path.join(args.private_model_path, filename), "w")
    args.log_file = file

    test_type = args.test_models_type
    # test_type = 'retrained'
    # test_type = 'private'
    if test_type == "private":
        args.save_model_path = args.private_model_path
    elif test_type == "retrained":
        args.save_model_path = args.retrained_private_model_path
    else:
        raise Exception(f"Unknown test_type: {test_type}")

    utils.augmented_print("##########################################", file)
    utils.augmented_print("Test models on '{}' dataset!".format(args.dataset),
                          file)
    utils.augmented_print(
        "Test models on '{}' architecture!".format(args.architecture), file
    )
    utils.augmented_print(
        "Number test models: {:d}".format(args.end_id - args.begin_id), file
    )
    if args.dataset == "pascal":
        evalloader = utils.load_evaluation_dataloader(args=args)
    else:
        evalloader = utils.load_unlabeled_dataloader(args=args)
    # evalloader = utils.load_private_data(args=args)[0]
    print(f"eval dataset: ", evalloader.dataset)

    if args.debug is True:
        # Logs about the eval set
        show_dataset_stats(
            dataset=evalloader.dataset, args=args, file=file,
            dataset_name="eval"
        )

    # Training
    summary = {
        metric.loss: [],
        metric.acc: [],
        metric.balanced_acc: [],
        metric.auc: [],
        metric.map: [],
    }
    for id in args.querying_parties:
        utils.augmented_print("##########################################",
                              file)

        model = load_private_model_by_id(
            args=args, id=id, model_path=args.save_model_path
        )

        result = eval_distributed_model(
            model=model, dataloader=evalloader, args=args)

        model_name = get_model_name_by_id(id=id)
        result["model_name"] = model_name
        result_str = from_result_to_str(result=result, sep="\n",
                                        inner_sep=args.sep)
        utils.print_metrics_detailed(results=result)
        utils.augmented_print(text=result_str, file=file, flush=True)
        summary = update_summary(summary=summary, result=result)

    utils.augmented_print("##########################################", file)

    for key, value in summary.items():
        if len(value) > 0:
            avg_value = np.mean(value)
            std_value = np.std(value)
            min_value = np.min(value)
            max_value = np.max(value)
            med_value = np.median(value)
            str_value = utils.get_value_str(value=np.array(value))
            utils.augmented_print(
                f"{key} of private models;average;{avg_value};std;{std_value};"
                f"min;{min_value};max;{max_value};median;{med_value};"
                f"value;{str_value}",
                file,
            )

    end_time = time.time()
    elapsed_time = end_time - start_time
    utils.augmented_print(f"elapsed time: {elapsed_time}\n", file, flush=True)
    utils.augmented_print("##########################################", file)
    file.close()


def main(args):
    # Random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # CUDA support
    args.cuda = torch.cuda.is_available()
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    set_dataset(args=args)

    for model in args.architectures:
        args.architecture = model
        print("architecture: ", args.architecture)
        # num_models_list = [5, 10, 20, 30, 50, 100]
        # num_models_list = [5, 10]
        # num_models_list = [1, 10, 20, 50]
        # num_models_list = [50]
        # num_models_list = [1, 10, 20, 50]
        # num_models_list = [50]
        # num_models_list = [20, 30, 50]
        num_models_list = [args.num_models]
        for num_models in num_models_list:
            print("num_models: ", num_models)
            args.num_models = num_models
            if len(num_models_list) > 1:
                # for running experiments with many number of models
                args.end_id = num_models

            architecture = args.architecture
            dataset = get_dataset_full_name(args=args)
            xray_views = "".join(args.xray_views)
            # Folders
            if args.use_pretrained_models:
                args.private_model_path = os.path.join(
                    "/home/nicolas/code/capc-learning",
                    "private-models",
                    dataset + "pre",
                    architecture,
                    "{:d}-models".format(args.num_models),
                    xray_views,
                )
            else:
                args.private_model_path = os.path.join(
                    "/home/nicolas/code/capc-learning",
                    "private-models",
                    dataset,
                    architecture,
                    "{:d}-models".format(args.num_models),
                    xray_views,
                )
            print("args.private_model_path: ", args.private_model_path)
            args.save_model_path = args.private_model_path
            if args.use_pretrained_models:
                args.ensemble_model_path = os.path.join(
                    args.path,
                    "ensemble-models",
                    dataset + "pre",
                    architecture,
                    "{:d}-models".format(args.num_models),
                    xray_views,
                )
            else:
                args.ensemble_model_path = os.path.join(
                    args.path,
                    "ensemble-models",
                    dataset,
                    architecture,
                    "{:d}-models".format(args.num_models),
                    xray_views,
                )

            args.non_private_model_path = os.path.join(
                args.path, "non-private-models", dataset, architecture
            )
            # dir = [args.mode, 'threshold:{:.1f}'.format(args.threshold), 'sigma-gnmax:{:.1f}'.format(args.sigma_gnmax),
            #        'sigma-threshold:{:.1f}'.format(args.sigma_threshold), 'budget:{:.2f}'.format(args.budget)]
            args.retrained_private_model_path = os.path.join(
                args.path,
                "retrained-private-models",
                dataset,
                architecture,
                "{:d}-models".format(args.num_models),
                args.mode,
                xray_views,
            )

            print(
                "args.retrained_private_models_path: ",
                args.retrained_private_model_path,
            )

            args.adaptive_model_path = os.path.join(
                args.path,
                "adaptive-model",
                dataset,
                architecture,
                "{:d}-models".format(args.num_models),
                args.mode,
                xray_views,
            )

            if args.attacker_dataset:
                args.adaptive_model_path = os.path.join(
                    args.path,
                    "adaptive-model",
                    dataset + "_" + args.attacker_dataset,
                    architecture,
                    "{:d}-models".format(args.num_models),
                    args.mode,
                    xray_views,
                )

            for path_name in [
                "private_model",
                "ensemble_model",
                "retrained_private_model",
                "adaptive_model",
            ]:
                path_name += "_path"
                args_path = getattr(args, path_name)
                # if os.path.exists(args_path):
                #     raise Exception(
                #         f'The {path_name}: {args_path} already exists.')
                # else:
                #     os.makedirs(args_path)
                if not os.path.exists(args_path):
                    os.makedirs(args_path)

            # for budget in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]:
            # for budget in [float('inf')]:
            # args.budget = args.budgets[0]
            # for budget in args.budgets:
            for private_tau in args.private_taus:
                args.private_tau = private_tau
                # for budget in [args.budget]:
                # for budget in [2.8]:
                args.budget = args.budgets[0]
                print("main budget: ", args.budget)
                for command in args.commands:
                    args.command = command
                    if command == "train_private_models":
                        train_private_models(args=args)
                    elif command == "evaluate_ensemble_model":
                        evaluate_ensemble_model(args=args)
                    elif command == "evaluate_big_ensemble_model":
                        evaluate_big_ensemble_model(args=args)
                    elif command == "query_ensemble_model":
                        if args.model_size == model_size.small:
                            query_ensemble_model(args=args)
                        elif args.model_size == model_size.big:
                            query_big_ensemble_model(args=args)
                        else:
                            raise Exception(
                                f"Unknown args.model_size: {args.model_size}."
                            )
                    elif command == "retrain_private_models":
                        retrain_private_models(args=args)
                    elif command == "train_student_model":
                        train_student_model(args=args)
                    elif command == "pytorch2pickle":
                        pytorch2pickle(args=args)
                    elif command == "test_models":
                        test_models(args=args)
                    elif command == "set_taus":
                        set_taus(args=args)
                    elif command == "train_model_adaptively":
                        train_model_adaptively(args=args)
                    elif command in [
                        "basic_model_stealing_attack",
                        "basic_model_stealing_attack_with_BO",
                    ]:
                        run_model_extraction(args=args)
                    elif command == "adaptive_queries_only":
                        run_model_extraction(args=args,
                                             no_model_extraction=True)
                    else:
                        raise Exception("Unknown command: {}".format(command))


if __name__ == "__main__":
    args = get_parameters()
    main(args)
