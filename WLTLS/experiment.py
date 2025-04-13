"""
Author: Itay Evron
See https://github.com/ievron/wltls/ for more technical and license information.
--------------------------------------------------------------------------------
See the paper for more details on the different losses.
"""

import numpy as np
from aux_lib import Timing, print_debug
import random

#########################################################################
# An entire experiment.
# Train and test model. Check binary losses.
#########################################################################
class Experiment():
    _model = None
    EPOCHS = None
    trainAccuracies = None
    validAccuracies = None
    trainTimes = None

    def __init__(self, model, epochs = 1):
        self._model = model
        self.EPOCHS = epochs
        self.trainAccuracies = []
        self.validAccuracies = []
        self.trainTimes = []

    # Test the model on a given set
    def _test(self, X, Y):
        t = Timing()

        yPredicted, extraOutput = self._model.predictFinal(X)

        accuracy = 100 * self.evaluateScore(Y, yPredicted)

        return { "accuracy": accuracy,
                 "time": t.get_elapsed_secs(),
                 "y_predicted": yPredicted,
                 "extra_output": extraOutput }


    # Return accuracy
    @staticmethod
    def evaluateScore(Y, predictedY):
        correct = 0

        for actual, predicted in zip(Y, predictedY):
            if actual == predicted:
                correct += 1

        return correct / len(Y)

    # Shuffle the given sets
    def _shuffleDataset(self, X, Y):
        c = list(zip(X, Y))

        random.shuffle(c)

        return zip(*c)

    def _runEpoch(self, id, Xtrain, Ytrain, Xvalid, Yvalid):
        Xtrain, Ytrain = self._shuffleDataset(Xtrain, Ytrain)

        print_debug("Train epoch {}/{}: ".format(id + 1, self.EPOCHS), end='', flush=True)

        t = Timing()
        t.start()

        yPredicted = self._model.train(Xtrain, Ytrain)

        trainAccuracy = 100 * self.evaluateScore(Ytrain, yPredicted)

        self.trainAccuracies.append(trainAccuracy)
        self.trainTimes.append(t.get_elapsed_secs())
        print("\033[92m{:.2f}%\033[0m in {}.\t".format(
            trainAccuracy, t.get_elapsed_time()), end='', flush=True)

        # Validation
        validRes = self._test(Xvalid, Yvalid)
        self.validAccuracies.append(validRes["accuracy"])
        print('Validation: \033[92m{:.2f}%\033[0m ({}).\t'.format(validRes["accuracy"], Timing.secondsToString(validRes["time"])), end='', flush=True)

        if id == 0 and validRes["extra_output"] is not None:
            print(validRes["extra_output"], end=' ', flush=True)

        return validRes

    def run(self, Xtrain, Ytrain, Xvalid, Yvalid, Xtest, Ytest,
            earlyStop=False,
            forceStop=False,
            modelLogPath=None,
            computeAverageLoss=True,
            returnBestValidatedModel = False):
        bestRes = 0

        if modelLogPath is not None:
            # If we need to save the model, we increase the recursion limit
            # so we can save objects like the tree in the code manager etc.
            # The code itself is not recursive!
            import sys
            sys.setrecursionlimit(10 ** 5)

        # Run training epochs
        epoch = 0
        requireAnotherEpoch = True
        notImproved = 0
        while epoch < self.EPOCHS or (requireAnotherEpoch and not forceStop):
            requireAnotherEpoch = False

            validRes = self._runEpoch(epoch, Xtrain, Ytrain, Xvalid, Yvalid, )

            if validRes["accuracy"] < bestRes:
                notImproved += 1

                # If didn't improve for two epochs => stop
                if notImproved == 2 and earlyStop:
                    print_debug("Early stopping!")
                    break
            # If our validation score is the highest so far
            else:
                notImproved = 0
                bestRes = validRes["accuracy"]
                requireAnotherEpoch = True

                # We save the model to file whenever we reach a better validation performance,
                # so that if the simulation will be terminated for some reason
                # (usually only happen when we didn't allocate enough space for the process)
                # we will have a backup.
                if modelLogPath is not None:
                    tSave = Timing()
                    tSave.start()

                    # Important:
                    # numpy.savez is originally meant for saving arrays, not objects,
                    # We use it here for simplicity but it sometimes causes very high additional memory requirements
                    # (it processes the data before saving).
                    #
                    # A better way would be to save the data of the binary learners (e.g. means of AROW),
                    # and the coding matrix and allocation, one by one, and then load them with a special method.
                    import os
                    os.makedirs(os.path.dirname(modelLogPath), exist_ok=True)
                    np.savez(modelLogPath, ltls=self._model)

                    print("Saved model ({}).".format(tSave.get_elapsed_time()), end='', flush=True)
                    del tSave

            print("")
            epoch += 1

        # If we need to return the best validated model, load it from file before continuing
        if returnBestValidatedModel:
            del self._model

            self._model = np.load(modelLogPath + ".npz", allow_pickle=True)["ltls"][()]

        # Test
        testRes = self._test(Xtest, Ytest)
        print_debug('Test accuracy: {:.1f}% ({})'.format(testRes["accuracy"], Timing.secondsToString(testRes["time"])))

        if computeAverageLoss:
            # Calculate average binary loss
            decodingLoss = self._calcBitwiseLoss(Xtrain, Ytrain, learners=30)
            print_debug("Average binary loss: {:,.2f}".format(decodingLoss))



    # Calculate the average binary loss (\epsilon in the paper)
    def _calcBitwiseLoss(self, X, Y, learners=None):
        lossFunc = self._model.loss

        if learners is None:
            loss = np.array([lossFunc(self._model._getMargins(x),
                                      self._model.codeManager.labelToCode(y))
                             for x,y in zip(X,Y)])
        else:
            loss = np.array([lossFunc(self._model._getPartialMargins(x, learners),
                                      self._model.codeManager.labelToCode(y)[:learners])
                             for x,y in zip(X,Y)])

        return np.mean(loss)

#########################################################################
# Explainable Experiment.
# Enhanced experiment with explainability and robustness metrics.
#########################################################################
class ExplainableExperiment(Experiment):
    def __init__(self, model, epochs=1):
        super().__init__(model, epochs)
        self.explanation_stats = {
            'robustness_scores': [],
            'confidence_scores': [],
            'edge_importance': {}
        }
        
    def _analyze_explanations(self, explanations):
        """Analyze explanations from a batch of predictions"""
        for exp in explanations:
            # Track robustness scores
            self.explanation_stats['robustness_scores'].append(
                exp['robustness']['robustness_score']
            )
            
            # Track confidence scores
            self.explanation_stats['confidence_scores'].append(
                exp['path_attribution']['decision_confidence']
            )
            
            # Aggregate edge importance
            for edge, importance in exp['path_attribution']['edge_importance'].items():
                if edge not in self.explanation_stats['edge_importance']:
                    self.explanation_stats['edge_importance'][edge] = []
                self.explanation_stats['edge_importance'][edge].append(importance)
    
    def _test(self, X, Y):
        """Enhanced testing with explainability and robustness metrics"""
        result = super()._test(X, Y)
        
        if hasattr(result, 'extra_output') and 'explanations' in result['extra_output']:
            self._analyze_explanations(result['extra_output']['explanations'])
            
            # Calculate aggregate statistics
            avg_robustness = np.mean(self.explanation_stats['robustness_scores'])
            avg_confidence = np.mean(self.explanation_stats['confidence_scores'])
            
            # Calculate most important edges
            edge_importance = {
                edge: np.mean(scores) 
                for edge, scores in self.explanation_stats['edge_importance'].items()
            }
            top_edges = sorted(
                edge_importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]
            
            result['explanation_metrics'] = {
                'average_robustness': avg_robustness,
                'average_confidence': avg_confidence,
                'top_important_edges': top_edges
            }
            
        return result

    def run(self, Xtrain, Ytrain, Xvalid, Yvalid, Xtest, Ytest,
            earlyStop=False, forceStop=False, modelLogPath=None,
            computeAverageLoss=True, returnBestValidatedModel=False):
        
        result = super().run(Xtrain, Ytrain, Xvalid, Yvalid, Xtest, Ytest,
                           earlyStop, forceStop, modelLogPath,
                           computeAverageLoss, returnBestValidatedModel)
        
        # Print explainability and robustness summary
        if hasattr(self, 'explanation_stats'):
            print("\nExplainability and Robustness Metrics:")
            print("-" * 40)
            
            if self.explanation_stats['robustness_scores']:
                print(f"Average Robustness Score: {np.mean(self.explanation_stats['robustness_scores']):.3f}")
                print(f"Average Confidence Score: {np.mean(self.explanation_stats['confidence_scores']):.3f}")
                
                print("\nTop 5 Most Important Decision Edges:")
                edge_importance = {
                    edge: np.mean(scores) 
                    for edge, scores in self.explanation_stats['edge_importance'].items()
                }
                for edge, importance in sorted(
                    edge_importance.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:5]:
                    print(f"Edge {edge}: {importance:.3f}")
                    
        return result