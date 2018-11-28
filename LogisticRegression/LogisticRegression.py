import math
import torch


class LogisticRegression(object):
    # TODO Add Init

    def __init__(self, theta_vector, alpha):
        self.theta_vector = theta_vector
        self.alpha = alpha

    def hypothesis(self, x_vector):
        """Calculates the sigmoid hypothesis for Logistic Regression

        ____1___________________
           -Theta_transverse_X
        1-e

        :param [tensors] x_vector: array of tensors representing the features of all data points.
        :return: hypothesis of thetas and features
        :rtype: [tensor]
        """
        hypothesis_vector = []
        for data_point in x_vector:
            theta_transverse_x = self._transverse(self.theta_vector, data_point)
            hypothesis_vector.append(torch.Tensor.item(torch.sigmoid(-theta_transverse_x, )))
        return torch.tensor(hypothesis_vector, dtype=torch.float)

    def _transverse(self, vector_A, vector_B):
        return torch.sum(torch.mul(vector_A, vector_B), dtype=torch.float)

    @staticmethod
    def binomial_hypothesis_quantization(hypothesis_array):
        binomial_hypothesis = []
        for hypothesis in hypothesis_array:
            if hypothesis >= 0.5:
                binomial_hypothesis.append(1)
            else:
                binomial_hypothesis.append(0)
        return binomial_hypothesis

    def cost(self, hypothesis_vector, actual_values_vector):
        '''Calculates the cost of the hypothesis and the actual values.

        For a good cost, it should return 0.5 or less. NaN is cost 0, and INF is cost infinity.
        :param tensor hypothesis_vector: vector that holds the hypothesis for given data points.
        :param tensor actual_values_vector: vector that holds actual values for given data point (0 or 1)
        :return: cost
        :rtype: tensor
        '''
        # Code essentially does the following:
        # cost = (1/X.size)((-yTransverseLogOfHypothesis)-(1-y)transverse(log(1-hypothesis)))

        one_over_m = -(1 / torch.numel(actual_values_vector))
        y_trans_logH = self._transverse(actual_values_vector, torch.log(hypothesis_vector))
        negY = torch.mul(actual_values_vector, -1)
        negH = torch.mul(hypothesis_vector, -1)
        one_min_Y_logH = self._transverse(torch.add(negY, 1), torch.log(torch.add(negH, 1)))

        cost = one_over_m * (y_trans_logH + one_min_Y_logH)
        print("Cost of hypothesis: {cost}".format(cost=cost))
        return cost

    def gradientDescent(self, hypothesis_vector, actual_value_vector, x_vector):
        """Runs one iteration of gradient descent for all thetas in the self.theta_vector

        Essentially, runs the following equation:
            theta_vector = theta_vector - (alpha/X.size)(x_vectorTransverse(hypothesisVector-actualresultVector))

        :param tensor hypothesis_vector: tensor holding all the hypothesis results for all data points.
        :param tensor actual_value_vector: tensor holding all actual values for all test data points.
        :param [tensor] x_vector: array of tensors holding all all features for all data points.
        :return: nothing; updates self.theta_vector
        """

        for data_point in x_vector:
            hypothesis_minus_y = torch.sum(torch.add( hypothesis_vector, -1, actual_value_vector))

            x_trans_HY = torch.mul(data_point, hypothesis_minus_y)

            alpha_over_m = self.alpha / torch.numel(actual_value_vector)
            RHS = torch.mul(x_trans_HY, alpha_over_m)

            # Changed sign of RHS here and seemed to work. Check up on this.
            self.theta_vector = torch.add(self.theta_vector, RHS)

    def accuracy(self, hypothesis, labels):
        max = len(hypothesis)
        i=0
        matches = 0
        while i<max:
            if hypothesis[i] == labels[i]:
                matches+=1
            i+=1
        return matches/max

# # Todo figure out how to fretunr/iterate over various data points.
# # Todo update code to work with multiple datapoints.````````
# data = [[1, 2, 2, 2, 2, 3],
#         [1, 1, 1, 1, 1, 3],
#         [1, 5, 5, 5, 5, 3],
#         [1, 9, 9, 9, 9, 3],
#         [1, 8, 8, 8, 8, 3], [1, 7, 7, 7, 7, 3],
#         [1, 0, 0, 0, 0, 3]
#         ]
# tensors = []
# for data_point in data:
#     test_X_Vector = torch.tensor(data_point, dtype=torch.float)
#     tensors.append(test_X_Vector)
#
# test_theta_Vector = torch.tensor([2, 2, 3, 2, 2, 3], dtype=torch.float)
#
# LR = LogisticRegression(test_theta_Vector, 0.00001)
# hypothesis = LR.hypothesis(tensors)
# print(hypothesis)
# realVal = torch.tensor([0, 0, 1, 1, 1, 1, 0], dtype=torch.float)
# #
# LR.cost(hypothesis, realVal)
# print(LR.theta_vector)
#
# while LR.cost(hypothesis, realVal) > 0.25:
#     LR.gradientDescent(hypothesis, realVal, test_X_Vector)
#     hypothesis = LR.hypothesis(tensors)
#     print(LR.theta_vector)
#
# LR.cost(hypothesis, realVal)
#
# hypothesis = LR.hypothesis(tensors)
# print(hypothesis.numpy())
# print(LR.binomial_hypothesis_quantization(hypothesis.numpy()))
