"""
Author: Itay Evron
See https://github.com/ievron/wltls/ for more technical and license information.
"""

import numpy as np
from aux_lib import print_debug
import numpy.linalg as la

#############################################################################################
# A W-LTLS model.
#############################################################################################

class WltlsModel():
    POTENTIAL_PATH_NUM = 20
    LABELS = 0
    DIM = 0
    codeManager = None
    _learners = None
    _decoder = None

    def __init__(self, LABELS, DIM, learnerClass, codeManager, decoder, isMultilabel = False,
                 step_size = None):
        if isMultilabel:
            raise NotImplementedError

        self.codeManager = codeManager
        learners = [learnerClass(DIM, step_size=step_size) for _ in range(self.codeManager._bitsNum)]

        self.LABELS = LABELS
        self.DIM = DIM
        self._learners = learners
        self._assignedAllLabels = False
        self._isMultilabel = isMultilabel

        self._decoder = decoder
        self.loss = decoder.loss
        print_debug("Model size: {0:,.1f}MB".format(self.getModelSize() / 1024 ** 2))

    def getActualLabels(self, y):
        return y.indices if self._isMultilabel else [y]

    def getModelSize(self):
        return sum([l.getModelSize() for l in self._learners])

    def getPredictorsNumber(self):
        return len(self._learners)

    # If we predict a yet-assigned code, we return None
    def _predictAndRefit(self, x, y):
        # Currently works only for multiclass! Multilabel require significant adjustments, see LTLS

        actualLabel = self.getActualLabels(y)[0]
        margins = self._getMargins(x)

        # It's a step towards multi-label support, but it doesn't work in this version.
        # In multi-class, there's only one positive (actual) label, and at most one negative (predicted) label
        actualCode = self.codeManager.labelToCode(actualLabel)

        # If the actual label isn't assigned yet to a path, find POTENTIAL_PATH_NUM heaviest paths to choose from
        if actualCode is None:
            actualCode = self.codeManager.assignLabel(actualLabel,
                                                      self._decoder.findKBestCodes(margins, self.POTENTIAL_PATH_NUM))


        # Learn independently
        #
        # Note about parallel learning:
        # This part could be made parallel with |learners| cores.
        # However, the code should be organized a little differently (the outer loop should be on cores/learners
        # and the inner one on the samples).
        # Moreover, to support the greedy path assignment method (see paper), we have to make a prediction for every
        # sample sequentially. The random path assignment policy fully support the learning in parallel.
        for i,l in enumerate(self._learners):
            y = actualCode[i]

            l.refit(x, y, margins[i])


        # Find best code (decoding)
        topCode = self._decoder.findKBestCodes(margins, 1)[0]

        # Find predicted label
        yPredicted = self.codeManager.codeToLabel(topCode)

        return yPredicted

    def train(self, X, Y):
        # Switches the decoding to train mode
        for l in self._learners:
            l.train()

        # Actually train (refit) the learners
        # (This part could be made parallel with |learners| cores)
        yPredicted = [self._predictAndRefit(x, y) for x,y in zip(X, Y)]

        return yPredicted

    # Predicts a sample's label.
    def _predict(self, x, k=1):
        margins = self._getMargins(x)

        codes = self._decoder.findKBestCodes(margins, k)

        return [self.codeManager.codeToLabel(code) for code in codes]

    # Predicts a sample's label.
    def predict(self, X):
        return np.array([self._predict(x) for x in X])

    # Compute the values of the binary functions on an input x
    def _getMargins(self, x):
        return [l.score(x) for l in self._learners]

    # Compute the values of the binary functions on an input x on a subset of predictors
    def _getPartialMargins(self, x, learners=1):
        return [l.score(x) for l in self._learners[:learners]]

    # Prepare for evaluation
    def eval(self):
        extraOutput = None

        # Before the first evaluation, assign all remaining codewords to paths.
        # Important in cases some class appear on the validation/test set but not on the train set.
        if not self._assignedAllLabels:
            extraOutput = self.prepareForFirstEvaluation()

            self._assignedAllLabels = True

        for l in self._learners:
            l.eval()

        return extraOutput

    # Predict many using the final model (after training)
    def predictFinal(self, X):
        extraOutput = self.eval()

        yPredicted = [self._predict(x) for x in X]

        return yPredicted, extraOutput

    # Called after the first training epoch, before the first evaluation.
    # Should return output to be printed, or None.
    def prepareForFirstEvaluation(self):
        # Assign codes to unassigned labels
        assigned = self.codeManager.assignRemainingCodes()

        return "Assigned {} labels.".format(assigned)


class ExplainableWltlsModel(WltlsModel):
    def __init__(self, LABELS, DIM, learnerClass, codeManager, decoder, isMultilabel=False, 
                 step_size=None, epsilon=0.1, interpret_k=5):
        super().__init__(LABELS, DIM, learnerClass, codeManager, decoder, isMultilabel, step_size)
        self.epsilon = epsilon  # Parameter for adversarial robustness
        self.interpret_k = interpret_k  # Number of top paths to consider for interpretation
        
    def get_path_attribution(self, x):
        """Calculate path attribution scores for interpretability"""
        margins = self._getMargins(x)
        top_k_codes = self._decoder.findKBestCodes(margins, self.interpret_k)
        
        # Calculate attribution scores for each edge
        edge_scores = {}
        for i, margin in enumerate(margins):
            edge_scores[i] = abs(margin)  # Importance of each edge decision
            
        # Get path probabilities
        path_probs = {}
        total_score = 0
        for code in top_k_codes:
            score = sum([margins[i] if c == 1 else -margins[i] for i, c in enumerate(code)])
            path_probs[tuple(code)] = np.exp(score)
            total_score += np.exp(score)
            
        # Normalize probabilities
        for k in path_probs:
            path_probs[k] /= total_score
            
        return {
            'edge_importance': edge_scores,
            'path_probabilities': path_probs,
            'decision_confidence': max(path_probs.values())
        }
        
    def verify_prediction_robustness(self, x, pred_label):
        """Verify if prediction is robust against adversarial perturbations"""
        margins = self._getMargins(x)
        orig_code = self.codeManager.labelToCode(pred_label)
        
        # Calculate decision boundary distance
        min_perturbation = float('inf')
        for label in range(self.LABELS):
            if label == pred_label:
                continue
                
            target_code = self.codeManager.labelToCode(label)
            if target_code is None:
                continue
                
            # Calculate minimum perturbation needed to change prediction
            diff_positions = [i for i in range(len(orig_code)) if orig_code[i] != target_code[i]]
            for pos in diff_positions:
                w = self._learners[pos].mean  # Get decision boundary normal vector
                margin = margins[pos]
                perturbation = abs(margin) / (la.norm(w) + 1e-10)
                min_perturbation = min(min_perturbation, perturbation)
        
        is_robust = min_perturbation > self.epsilon
        return {
            'is_robust': is_robust,
            'perturbation_bound': min_perturbation,
            'robustness_score': min_perturbation / self.epsilon
        }
        
    def _predictAndRefit(self, x, y):
        """Enhanced prediction with both explainability and robustness"""
        pred_label = super()._predictAndRefit(x, y)
        
        if pred_label is not None:
            # Get explainability metrics
            path_attrs = self.get_path_attribution(x)
            
            # Check robustness
            robust_metrics = self.verify_prediction_robustness(x, pred_label)
            
            # Store metrics for later use
            self._last_explanation = {
                'path_attribution': path_attrs,
                'robustness': robust_metrics
            }
            
        return pred_label

    def predictFinal(self, X):
        """Enhanced final prediction with explanations and robustness checks"""
        yPredicted, extraOutput = super().predictFinal(X)
        
        explanations = []
        for i, x in enumerate(X):
            # Extract the first prediction since _predict returns a list
            pred = yPredicted[i][0] if isinstance(yPredicted[i], list) else yPredicted[i]
            
            path_attrs = self.get_path_attribution(x)
            robust_metrics = self.verify_prediction_robustness(x, pred)
            
            explanations.append({
                'path_attribution': path_attrs,
                'robustness': robust_metrics
            })
            
        return yPredicted, {
            'base_output': extraOutput,
            'explanations': explanations
        }