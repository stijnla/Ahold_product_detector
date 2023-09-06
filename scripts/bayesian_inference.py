import numpy as np

class BayesianInference():
    
    def __init__(self, num_classes):
        self._num_classes = num_classes
        self._probability_a_priori = (1/num_classes)*np.ones((num_classes, 1)).T # All classes start with equal probability of occuring


    def update(self, probability_detection):
        print()
        if np.sum(probability_detection) != 1:
            raise ValueError("The detection does not have a set of probabilities that sum to one")
        
        if hasattr(self, '_probability_posteriori'):
            self._probability_a_priori = self._probability_posteriori # if already received a detection, update the a_priori knowledge

        print(self._probability_a_priori)
        print(probability_detection)
        self._probability_posteriori = probability_detection * self._probability_a_priori # not normalized (sum != 1)
        print(self._probability_posteriori)
        self._probability_posteriori = self._probability_posteriori/self._probability_posteriori.sum(1) # now its normalized
        print(self._probability_posteriori)
        
    @property
    def classification(self):
        _most_likely_class = np.argmax(self._probability_posteriori)
        return self._probability_posteriori[_most_likely_class], _most_likely_class
    
classifications = BayesianInference(3)

detection1 = np.array([0.8, 0.1, 0.1])
detection2 = np.array([0.4, 0.2, 0.4])

classifications.update(detection1)
print(classifications.classification)
classifications.update(detection1)
print(classifications.classification)
classifications.update(detection2)
print(classifications.classification)
