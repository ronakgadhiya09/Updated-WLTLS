"""
Author: Itay Evron
See https://github.com/ievron/wltls/ for more technical and license information.
"""

import numpy as np
from aux_lib import Timing

#############################################################################################
# A model for fast inference.
# Merges separate previously-learned weight vectors into on weights matrix.
# Note that we learn the vectors separately to support parallel learning and/or
# the greedy path-to-label allocation policy.
#############################################################################################
class FinalModel:
    def __init__(self, DIMS, trainedModel, codeManager, decoder):
        print("Preparing a final model (this may take some time)...")

        # Merge trained model into one matrix for faster inference
        # (maybe could be done more efficiently in terms of memory)
        self.W = np.ravel(np.array([l.mean for l in trainedModel._learners]).T).reshape((DIMS, -1))

        # Delete the separate vectors from the memory
        for learner in trainedModel._learners:
            del learner

        self.codeManager = codeManager
        self.decoder = decoder

        print("The final model created successfully.")

    def _get_class_scores(self, responses):
        # Get all possible codes and their scores
        codes = self.decoder.allCodes()
        scores = {}
        
        # For each class, find its code and corresponding score
        for label in range(self.codeManager.LABELS):
            code = self.codeManager.labelToCode(label)
            if code is not None:
                # Compute score for this class based on its code
                score = sum([r if c == 1 else -r for r, c in zip(responses, code)])
                scores[label] = score
        
        return scores

    # Test the final model
    def test(self, Xtest, Ytest, save_scores=True, scores_file=None):
        t = Timing()
        Ypredicted = [0] * Xtest.shape[0]
        all_scores = []

        print("Computing confidence scores for {} test samples...".format(Xtest.shape[0]))
        
        for i, x in enumerate(Xtest):
            # Get responses from predictors
            responses = x.dot(self.W).ravel()

            # Get scores for all classes
            scores = {}
            for label in range(self.codeManager.LABELS):
                code = self.codeManager.labelToCode(label)
                if code is not None:
                    score = sum([r if c == 1 else -r for r, c in zip(responses, code)])
                    scores[label] = score
            
            all_scores.append(scores)

            # Print scores for first few examples
            if i < 3:  # Show first 3 examples
                print("\nSample {}: ".format(i))
                sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                for label, score in sorted_scores[:5]:  # Show top 5 classes
                    print("Class {}: {:.4f}".format(label, score))

            # Find best code using graph inference
            code = self.decoder.findKBestCodes(responses, 1)[0]
            Ypredicted[i] = self.codeManager.codeToLabel(code)

        if save_scores:
            if scores_file is None:
                scores_file = "confidence_scores.txt"
            
            try:
                with open(scores_file, 'w') as f:
                    f.write("Sample_ID\t" + "\t".join(f"Class_{i}" for i in range(self.codeManager.LABELS)) + "\n")
                    for i, scores in enumerate(all_scores):
                        score_values = [str(scores.get(j, float('-inf'))) for j in range(self.codeManager.LABELS)]
                        f.write(f"{i}\t" + "\t".join(score_values) + "\n")
                print("\nConfidence scores successfully saved to: {}".format(scores_file))
            except Exception as e:
                print("\nWarning: Failed to save confidence scores to file: {}".format(str(e)))

        correct = sum([y1 == y2 for y1, y2 in zip(Ypredicted, Ytest)])
        accuracy = correct * 100 / Xtest.shape[0]

        elapsed = t.get_elapsed_secs()
        print("\nTesting completed in {:.2f} seconds".format(elapsed))
        
        return {
            "accuracy": accuracy,
            "time": elapsed,
            "y_predicted": Ypredicted,
            "all_scores": all_scores 
        }
