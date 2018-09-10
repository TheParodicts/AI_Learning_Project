test_features = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
test_results = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
theta0 = 7
theta1 = 6
alpha = 0.01


class Data(object):
    """An object to hold the required dataset data."""

    def __init__(self, features, results, theta0=0, theta1=1, alpha=0.001):
        self.features = features
        self.results = results
        self.theta0 = theta0
        self.theta1 = theta1
        self.alpha = alpha

    def print_info(self):
        print("Features:" + self.features.__str__())
        print("Results:" + self.results.__str__())
        print("Theta0 =" + self.theta0.__str__())
        print("Theta1 =" + self.theta1.__str__())
        print(self.cost())

    def cost(self):
        print("COST METHOD:")
        sum_right = 0
        n = 0

        while n < len(self.features):
            sum_right = sum_right + (
                pow(self.hypothesis_minus_actual_value(feature=self.features[n], actual_value=self.results[n]), 2))
            n = n + 1
        cost_result = sum_right / (2 * (len(self.features)))

        return cost_result

    def hypothesis(self, feature):
        return self.theta0 + (self.theta1 * feature)

    def hypothesis_minus_actual_value(self, feature, actual_value):
        return self.hypothesis(feature=feature) - actual_value

    def gradient_descent_0(self, theta0):
        sum_right = 0
        n = 0

        while n < len(self.features):
            sum_right = sum_right + self.hypothesis_minus_actual_value(feature=self.features[n],
                                                                       actual_value=self.results[n])
            n = n + 1
        sum_right = sum_right * alpha
        sum_right = sum_right / len(self.features)

        return theta0 - sum_right

    def gradient_descent_1(self, theta1):
        sum_right = 0
        n = 0

        while n < len(test_features):
            sum_right = sum_right + (
                    self.hypothesis_minus_actual_value(feature=self.features[n], actual_value=self.results[n]) *
                    self.features[n])
            n = n + 1
        sum_right = sum_right * alpha
        sum_right = sum_right / len(self.features)

        return theta1 - sum_right

    def batch_gradient_descent(self, threshold=0.01):

        lim = 100000
        old_theta0 = self.theta0
        old_theta1 = self.theta1
        has_not_run_yet = True
        counter = 0
        while counter < lim:
            temp0 = self.gradient_descent_0(self.theta0)
            temp1 = self.gradient_descent_1(self.theta1)
            old_theta0 = self.theta0
            old_theta1 = self.theta1
            self.theta0 = temp0
            self.theta1 = temp1
            has_not_run_yet = False
            counter += 1

        print("New Thetas: " + self.theta0.__str__() + ", " + self.theta1.__str__())

# test_data_object = Data(features=test_features, results=test_results, theta0=theta0, theta1=theta1, alpha=alpha)
# test_data_object.print_info()
# test_data_object.batch_gradient_descent()
# test_data_object.print_info()
