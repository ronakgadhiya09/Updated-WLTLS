from WLTLS.datasets import read
from WLTLS.decoding import HeaviestPaths, expLoss, squaredLoss, squaredHingeLoss
from WLTLS.mainModels.finalModel import FinalModel
from graphs import TrellisGraph
import argparse
import warnings
import numpy as np
from WLTLS.mainModels.WltlsModel import ExplainableWltlsModel
from WLTLS.learners import AveragedPerceptron, AROW
from aux_lib import Timing, print_debug
from WLTLS.codeManager import GreedyCodeManager, RandomCodeManager
from WLTLS.datasets import datasets
from WLTLS.experiment import ExplainableExperiment
import os

# Constants
LEARNER_AROW = "AROW"
LEARNER_PERCEPTRON = "perceptron"
LOSS_EXP = "exponential"
LOSS_SQUARED = "squared"
LOSS_SQUARED_HINGE = "squared_hinge"
ASSIGNMENT_GREEDY = "greedy"
ASSIGNMENT_RANDOM = "random"

def run_explainable_experiment(dataset_name, data_path, model_dir, 
                             slice_width=5, epsilon=0.1, interpret_k=5,
                             rnd_seed=None, decoding_loss=LOSS_EXP,
                             path_assignment=ASSIGNMENT_RANDOM,
                             binary_classifier=LEARNER_AROW):
    
    if rnd_seed is not None:
        import random
        random.seed(rnd_seed)
        np.random.seed(rnd_seed)

    # Get dataset parameters
    DATASET = datasets.getParams(dataset_name)
    EPOCHS = DATASET.epochs
    LOG_PATH = os.path.join(model_dir, "model")

    print("=" * 80)
    print("Learning an Explainable Wide-LTLS model")
    print("=" * 80)

    # Load dataset
    Xtrain, Ytrain, Xvalid, Yvalid, Xtest, Ytest, LABELS, DIMS = read(data_path, DATASET)

    print("=" * 80)

    # Set up the loss function
    if decoding_loss == LOSS_EXP:
        loss = expLoss
    elif decoding_loss == LOSS_SQUARED_HINGE:
        loss = squaredHingeLoss
    else:
        loss = squaredLoss

    # Create graph and decoder
    trellisGraph = TrellisGraph(LABELS, slice_width)
    heaviestPaths = HeaviestPaths(trellisGraph, loss=loss)

    # Set up code manager
    if path_assignment == ASSIGNMENT_RANDOM:
        codeManager = RandomCodeManager(LABELS, heaviestPaths.allCodes())
    else:
        codeManager = GreedyCodeManager(LABELS, heaviestPaths.allCodes())

    # Set up learner
    learner = AROW if binary_classifier == LEARNER_AROW else AveragedPerceptron

    print(f"Using {binary_classifier} as the binary classifier.")
    print(f"Decoding according to the {decoding_loss} loss.")
    print(f"Adversarial robustness epsilon: {epsilon}")
    print(f"Number of paths for interpretation: {interpret_k}")

    # Create explainable model
    mainModel = ExplainableWltlsModel(
        LABELS, DIMS, learner, codeManager, heaviestPaths,
        epsilon=epsilon, interpret_k=interpret_k
    )

    print("=" * 80)

    # Run explainable experiment
    ExplainableExperiment(mainModel, EPOCHS).run(
        Xtrain, Ytrain, Xvalid, Yvalid, Xtest, Ytest,
        modelLogPath=LOG_PATH,
        returnBestValidatedModel=True
    )

    print("=" * 80)

    # Create final model with explainability
    finalModel = FinalModel(DIMS, mainModel, codeManager, heaviestPaths)
    del mainModel

    result = finalModel.test(Xtest, Ytest)
    print(f"Final model accuracy: {result['accuracy']:.1f}% ({Timing.secondsToString(result['time'])})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runs an Explainable W-LTLS experiment.")
    
    parser.add_argument("dataset", choices=[d.name for d in datasets.getAll()],
                        help="Dataset name")
    parser.add_argument("data_path", help="Path of the datasets directory")
    parser.add_argument("model_dir", help="Path to save the model")
    parser.add_argument("-slice_width", type=int, default=5,
                        help="Width of slices in the trellis graph")
    parser.add_argument("-epsilon", type=float, default=0.1,
                        help="Adversarial robustness threshold")
    parser.add_argument("-interpret_k", type=int, default=5,
                        help="Number of top paths to consider for interpretation")
    parser.add_argument("-rnd_seed", type=int, help="Random seed")
    parser.add_argument("-decoding_loss", 
                        choices=[LOSS_EXP, LOSS_SQUARED, LOSS_SQUARED_HINGE],
                        default=LOSS_EXP,
                        help="Decoding loss function")
    parser.add_argument("-path_assignment",
                        choices=[ASSIGNMENT_RANDOM, ASSIGNMENT_GREEDY],
                        default=ASSIGNMENT_RANDOM,
                        help="Path assignment policy")
    parser.add_argument("-binary_classifier",
                        choices=[LEARNER_AROW, LEARNER_PERCEPTRON],
                        default=LEARNER_AROW,
                        help="Binary classifier type")
    
    args = parser.parse_args()
    
    run_explainable_experiment(
        args.dataset,
        args.data_path,
        args.model_dir,
        args.slice_width,
        args.epsilon,
        args.interpret_k,
        args.rnd_seed,
        args.decoding_loss,
        args.path_assignment,
        args.binary_classifier
    )