# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Provides black-box gradient estimation using NES.
"""
import logging
from typing import List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
from scipy.stats import entropy

from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassifierMixin, ClassifierLossGradients
from art.utils import clip_and_round

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE

logger = logging.getLogger(__name__)

import itertools

class QueryEfficientGradientEstimationClassifier(ClassifierLossGradients, ClassifierMixin, BaseEstimator):
    """
    Implementation of Query-Efficient Black-box Adversarial Examples. The attack approximates the gradient by
    maximizing the loss function over samples drawn from random Gaussian noise around the input.

    | Paper link: https://arxiv.org/abs/1712.07113
    """

    estimator_params = ["num_basis", "sigma", "round_samples"]

    def __init__(
        self,
        classifier: "CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE",
        num_basis: int,
        sigma: float,
        round_samples: float = 0.0,
    ) -> None:
        """
        :param classifier: An instance of a classification estimator whose loss_gradient is being approximated.
        :param num_basis:  The number of samples to draw to approximate the gradient.
        :param sigma: Scaling on the Gaussian noise N(0,1).
        :param round_samples: The resolution of the input domain to round the data to, e.g., 1.0, or 1/255. Set to 0 to
                              disable.
        """
        super().__init__(model=classifier.model, clip_values=classifier.clip_values)
        # pylint: disable=E0203
        self._classifier = classifier
        self.num_basis = num_basis
        self.sigma = sigma
        self.round_samples = round_samples
        self._nb_classes = self._classifier.nb_classes

    @property
    def input_shape(self) -> Tuple[int, ...]:
        """
        Return the shape of one input sample.

        :return: Shape of one input sample.
        """
        return self._classifier.input_shape  # type: ignore

    def predict(self, x: np.ndarray, batch_size: int = 128, **kwargs) -> np.ndarray:  # pylint: disable=W0221
        """
        Perform prediction of the classifier for input `x`. Rounds results first.

        :param x: Features in array of shape (nb_samples, nb_features) or (nb_samples, nb_pixels_1, nb_pixels_2,
                  nb_channels) or (nb_samples, nb_channels, nb_pixels_1, nb_pixels_2).
        :param batch_size: Size of batches.
        :return: Array of predictions of shape `(nb_inputs, nb_classes)`.
        """
        return self._classifier.predict(clip_and_round(x, self.clip_values, self.round_samples), batch_size=batch_size)

    def fit(self, x: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """
        Fit the classifier using the training data `(x, y)`.

        :param x: Features in array of shape (nb_samples, nb_features) or (nb_samples, nb_pixels_1, nb_pixels_2,
                  nb_channels) or (nb_samples, nb_channels, nb_pixels_1, nb_pixels_2).
        :param y: Target values (class labels in classification) in array of shape (nb_samples, nb_classes) in
                  one-hot encoding format.
        :param kwargs: Dictionary of framework-specific arguments.
        """
        raise NotImplementedError

    def _generate_samples(self, x: np.ndarray, epsilon_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate samples around the current image.

        :param x: Sample input with shape as expected by the model.
        :param epsilon_map: Samples drawn from search space.
        :return: Two arrays of new input samples to approximate gradient.
        """
        minus = clip_and_round(
            np.repeat(x, self.num_basis, axis=0) - epsilon_map,
            self.clip_values,
            self.round_samples,
        )
        plus = clip_and_round(
            np.repeat(x, self.num_basis, axis=0) + epsilon_map,
            self.clip_values,
            self.round_samples,
        )
        return minus, plus

    def class_gradient(self, x: np.ndarray, label: Union[int, List[int], None] = None, **kwargs) -> np.ndarray:
        """
        Compute per-class derivatives w.r.t. `x`.

        :param x: Input with shape as expected by the classifier's model.
        :param label: Index of a specific per-class derivative. If an integer is provided, the gradient of that class
                      output is computed for all samples. If multiple values as provided, the first dimension should
                      match the batch size of `x`, and each value will be used as target for its corresponding sample in
                      `x`. If `None`, then gradients for all classes will be computed for each sample.
        :return: Array of gradients of input features w.r.t. each class in the form
                 `(batch_size, nb_classes, input_shape)` when computing for all classes, otherwise shape becomes
                 `(batch_size, 1, input_shape)` when `label` parameter is specified.
        """
        raise NotImplementedError

    def _generate_sample_i(self, x: np.ndarray, epsilon_map: np.ndarray, i: int) -> Tuple[np.ndarray, np.ndarray]:
        minus = clip_and_round(
            x - epsilon_map[i],
            self.clip_values,
            self.round_samples,
        )
        plus = clip_and_round(
            x + epsilon_map[i],
            self.clip_values,
            self.round_samples,
        )
        return minus, plus

    def loss_gradient(self, x: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        if self.amortized_attack:
            return self.loss_gradient_new_efficient(x, y)
            #return self.loss_gradient_new(x, y)
        else:
            return self.loss_gradient_old(x, y)

        #return self.loss_gradient_new(x, y)
        #return self.loss_gradient_new_efficient(x, y)

    def loss_gradient_new_efficient(self, x: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute the gradient of the loss function w.r.t. `x`.

        :param x: Sample input with shape as expected by the model.
        :param y: Correct labels, one-vs-rest encoding.
        :return: Array of gradients of the same shape as `x`.
        """
        epsilon_map = self.sigma * np.random.normal(size=([self.num_basis] + list(self.input_shape)))
        #print(epsilon_map.shape)
        #print(epsilon_map.reshape(self.num_basis, -1).shape)
        grads = [0.0] * len(x)

        #print('eps map shape', epsilon_map.shape)
        #print('epsmap 11', epsilon_map[11])

        #batch over multiple examples
        reps_per_batch = 10
        reps = epsilon_map.shape[0]
        for jb in range(0, reps, reps_per_batch):
            minus_preds = []
            len_x = len(x)
            pm_len = 2*len_x*reps_per_batch
            minuses_pluses = [None]*pm_len

            for b in range(reps_per_batch):
              j = jb + b
              #print('j', j, 'b', b)
              if j >= reps:
                  b -= 1
                  #print('b after dec', b)
                  break

              for i in range(len(x)):
                minus, plus = self._generate_sample_i(x[i : i + 1], epsilon_map, j)
                #print('j', j)
                #print('minus i', i + b*2*len_x, 'plus i', i + len_x + b*2*len_x)
                minuses_pluses[i + b*2*len_x] = minus
                minuses_pluses[i + len_x + b*2*len_x] = plus
                
            #print('b after loop', b)
            if jb + reps_per_batch > reps:
                #print(minuses_pluses[:(b+1)*2*len_x])
                #print(minuses_pluses[(b+1)*2*len_x:])
                minuses_pluses = minuses_pluses[:(b+1)*2*len_x]
            

            #print('len(minuses_pluses)', len(minuses_pluses))

            minuses_pluses = np.array(minuses_pluses)
            minuses_pluses = np.squeeze(minuses_pluses, 1)

            #print(minuses_pluses.shape)
            pm_preds = self.predict(minuses_pluses, batch_size=4000)
            #minus_preds, plus_preds  = np.split(pm_preds, 2)

            #print('num pm preds', pm_preds.shape)
            #print('b', b+1)
            rounds = np.split(pm_preds, b+1)
            #print('len(rounds)', len(rounds))
            for rn, r in enumerate(rounds):
              minus_preds, plus_preds  = np.split(r, 2)
              #print(minus_preds.shape, plus_preds.shape)
              j = jb + rn

              for i, (mp, pp) in enumerate(zip(minus_preds, plus_preds)):
                new_y_minus = entropy(y[i], mp)
                new_y_plus = entropy(y[i], pp)

                one_grad = epsilon_map[j] * (new_y_plus - new_y_minus)
                grads[i] += one_grad

        for i in range(len(grads)):
            grads[i] = grads[i] / (self.num_basis * self.sigma)

        grads = self._apply_preprocessing_gradient(x, np.array(grads))
        return grads


    def loss_gradient_new(self, x: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute the gradient of the loss function w.r.t. `x`.

        :param x: Sample input with shape as expected by the model.
        :param y: Correct labels, one-vs-rest encoding.
        :return: Array of gradients of the same shape as `x`.
        """
        epsilon_map = self.sigma * np.random.normal(size=([self.num_basis] + list(self.input_shape)))
        #print(epsilon_map.shape)
        #print(epsilon_map.reshape(self.num_basis, -1).shape)
        grads = [0.0] * len(x)

        for j in range(epsilon_map.shape[0]):
            minus_preds = []
            #plus_preds = []
            pluses = []
            minus = None
            plus = None
            for r in range(2):
                for i in range(len(x)):
                    if r == 0:
                        minus, plus = self._generate_sample_i(x[i : i + 1], epsilon_map, j)
                        minus_preds.append(self.predict(minus)[0])
                        pluses.append(plus)
                    else:
                        plus_pred = self.predict(pluses[i])[0]
                        
                        new_y_minus = entropy(y[i], minus_preds[i])
                        new_y_plus = entropy(y[i], plus_pred)

                        one_grad = epsilon_map[j] * (new_y_plus - new_y_minus)
                        grads[i] += one_grad

        #for j in range(epsilon_map.shape[0]):
        #    for i in range(len(x)):
        #        minus, plus = self._generate_sample_i(x[i : i + 1], epsilon_map, j)
        #        pred = self.predict(np.concatenate((minus, plus)))

        #        new_y_minus = entropy(y[i], pred[0])
        #        new_y_plus = entropy(y[i], pred[1])

        #        one_grad = epsilon_map[j] * (new_y_plus - new_y_minus)
        #        grads[i] += one_grad

        #    #pluses = [self._generate_sample_i(x[i : i + 1], epsilon_map, j)[1][0] for i in range(len(x))]
        #    #plus_preds = self.predict(pluses)
        #    #print('plus_preds.shape', plus_preds.shape)
        #    #print(len(pluses))

        #    #minuses = [self._generate_sample_i(x[i : i + 1], epsilon_map, j)[0][0] for i in range(len(x))]
        #    #minus_preds = self.predict(minuses)
        #    #print('minus_preds.shape', minus_preds.shape)

        #    #for i in range(len(x)):
        #    #    grads[i] += epsilon_map[j] * (plus_preds[i] - minus_preds[i])

        for i in range(len(grads)):
            grads[i] = grads[i]* 2/self.num_basis / (2 * self.sigma)

        grads = self._apply_preprocessing_gradient(x, np.array(grads))
        return grads

    def loss_gradient_old(self, x: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        #new_grads = self.loss_gradient_new(x, y)
        """
        Compute the gradient of the loss function w.r.t. `x`.

        :param x: Sample input with shape as expected by the model.
        :param y: Correct labels, one-vs-rest encoding.
        :return: Array of gradients of the same shape as `x`.
        """
        epsilon_map = self.sigma * np.random.normal(size=([self.num_basis] + list(self.input_shape)))
        #print(epsilon_map.shape)
        #print(epsilon_map.reshape(self.num_basis, -1).shape)
        grads = []
        for i in range(len(x)):
            #print('i', i)
            minus, plus = self._generate_samples(x[i : i + 1], epsilon_map)
            #print('shape', minus.shape, plus.shape)

            # Vectorized; small tests weren't faster
            # ent_vec = np.vectorize(lambda p: entropy(y[i], p), signature='(n)->()')
            # new_y_minus = ent_vec(self.predict(minus))
            # new_y_plus = ent_vec(self.predict(plus))
            # Vanilla
            new_y_minus = np.array([entropy(y[i], p) for p in self.predict(minus, batch_size=4000)])
            new_y_plus = np.array([entropy(y[i], p) for p in self.predict(plus, batch_size=4000)])

            #print('term1 shape', epsilon_map.reshape(self.num_basis, -1).shape)
            #print('term2 shape', ((new_y_plus - new_y_minus).reshape(self.num_basis, -1) / (2 * self.sigma)).shape)

            query_efficient_grad = 2 * np.mean(
                np.multiply(
                    epsilon_map.reshape(self.num_basis, -1),
                    (new_y_plus - new_y_minus).reshape(self.num_basis, -1) / (2 * self.sigma),
                ).reshape([-1] + list(self.input_shape)),
                axis=0,
            )
            grads.append(query_efficient_grad)
        grads = self._apply_preprocessing_gradient(x, np.array(grads))
        #print('old grads', grads)
        #print('new grads', new_grads)
        #print('equal', grads == new_grads)
        return grads

    def get_activations(self, x: np.ndarray, layer: Union[int, str], batch_size: int) -> np.ndarray:
        """
        Return the output of the specified layer for input `x`. `layer` is specified by layer index (between 0 and
        `nb_layers - 1`) or by name. The number of layers can be determined by counting the results returned by
        calling `layer_names`.

        :param x: Input for computing the activations.
        :param layer: Layer for computing the activations.
        :param batch_size: Size of batches.
        :return: The output of `layer`, where the first dimension is the batch size corresponding to `x`.
        """
        raise NotImplementedError

    def save(self, filename: str, path: Optional[str] = None) -> None:
        """
        Save a model to file specific to the backend framework.

        :param filename: Name of the file where to save the model.
        :param path: Path of the directory where to save the model. If no path is specified, the model will be stored in
                     the default data location of ART at `ART_DATA_PATH`.
        """
        raise NotImplementedError
