class Data(object):
    """An object to hold the required dataset data."""

    def __init__(self, features, results, theta0=0, theta1=1, alpha=0.00001):
        """Initialize data, hypothesis, and Gradient descent variables
        """

        self.features = features
        self.results = results
        self.theta0 = theta0
        self.theta1 = theta1
        self.alpha = alpha

    def print_info(self):
        """Prints Current Data Info
        """

        print("Features:" + self.features.__str__())
        print("Results:" + self.results.__str__())
        print("Theta0 =" + self.theta0.__str__())
        print("Theta1 =" + self.theta1.__str__())
        print(self.cost())

    def cost(self):
        """Prints the current cost of the hypothesis.
        """

        print("COST METHOD:")
        sum_right = 0
        n = 0
        length_of_features = len(self.features)

        # The sum of the right hand side of the Cost function:
        # sum of the square of the hypothesis of x sub i minus the actual value for x sub i, for all i in the features.
        while n < length_of_features:
            sum_right += pow(self.hypothesis_minus_actual_value(feature=self.features[n],
                                                                actual_value=self.results[n]), 2)
            n += 1

        # Divide the sum by twice the length of features.
        cost_result = sum_right / (2 * length_of_features)

        return cost_result

    def hypothesis(self, feature):
        """Calculate the guess based on the current thetas.

        :param float feature: variable used to calculate guess.
        :return: float guess: the result of the hypothesis
        """
        return self.theta0 + (self.theta1 * feature)

    def hypothesis_minus_actual_value(self, feature, actual_value):
        """Subtracts the actual value for a feature from the hypothesis calculated result of said feature.

        :param float feature: feature tu use.
        :param float actual_value: actual value corresponding to feature
        :return: hypothesis result minus actual value
        :rtype: float
        """
        return self.hypothesis(feature=feature) - actual_value

    def gradient_descent_0(self):
        """Calculates a single step in gradient descent for theta 0.

        :return: gradient descent result for theta 0.
        :rtype: float
        """

        sum_right = 0
        n = 0

        # Sumation of the hypothesis minus actual value for every feature in data set.
        while n < len(self.features):
            sum_right += self.hypothesis_minus_actual_value(feature=self.features[n],
                                                            actual_value=self.results[n])
            n = n + 1

        # Multiply sum by alpha and divide by number of data elements.
        sum_right = (sum_right * self.alpha) / len(self.features)

        # Return old theta 0 minus right hand side of gradient descent.
        return self.theta0 - sum_right

    def gradient_descent_1(self):
        """Calculates a single step in gradient descent for theta 1.

        :return: gradient descent result for theta 1.
        :rtype: float
        """
        sum_right = 0
        n = 0

        # Sumation of the hypothesis minus actual value for every feature in data set times the feature.
        while n < len(self.features):
            sum_right += self.hypothesis_minus_actual_value(feature=self.features[n],
                                                            actual_value=self.results[n]) * self.features[n]
            n = n + 1

        # Multiply sum by alpha and divide by number of data elements.
        sum_right = sum_right * self.alpha / len(self.features)

        # Return old theta 0 minus right hand side of gradient descent.
        return self.theta1 - sum_right

    def batch_gradient_descent(self, threshold=0.00001):
        """Calculates batch gradient descent for both thetas to a certain degree of precision.

        Calculates the batch gradient descent for both  theta 0 and theta 1, then updates them
            simultaneously, until the difference is less than a threshold.

        :param float threshold: threshold value for which difference between new theta and old theta should be
            less than.
        """

        old_theta0 = self.theta0
        old_theta1 = self.theta1
        has_not_run_yet = True
        while has_not_run_yet or (
                (abs(old_theta0 - self.theta0) > threshold) or
                (abs(old_theta1 - self.theta1) > threshold)):
            temp0 = self.gradient_descent_0()
            temp1 = self.gradient_descent_1()
            old_theta0 = self.theta0
            old_theta1 = self.theta1
            self.theta0 = temp0
            self.theta1 = temp1
            has_not_run_yet = False

        print("New Thetas: " + self.theta0.__str__() + ", " + self.theta1.__str__())