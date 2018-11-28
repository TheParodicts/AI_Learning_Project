from LogisticRegression import LogisticRegression
import csv
import torch
import numpy as np

with open("candy-data.csv", "r") as csv_doc:
    reader = csv.reader(csv_doc, delimiter=",", quotechar='"')
    data_names = []
    data_labels = []
    data_features = []
    first_run = True
    for row in reader:
        #Ignore header row.
        if first_run:
            first_run = False
        else:
            data_names.append(row[0])
            data_labels.append(float(row[1]))
            features = row[2:4]
            # features.insert(0,1)
            feat = []
            for number in features:
                feat.append(float(number))
            data_features.append(torch.tensor(feat, dtype=torch.float))

theta_vector =  torch.tensor([1,1], dtype = torch.float)
data_labels = torch.tensor(data_labels, dtype=torch.float)

candy_LR = LogisticRegression(theta_vector, 0.00001)

hypothesis = candy_LR.hypothesis(data_features)
print(hypothesis)

cost = candy_LR.cost(hypothesis, data_labels)

while candy_LR.cost(hypothesis, data_labels) > .6:
    candy_LR.gradientDescent(hypothesis, data_labels, data_features)
    hypothesis = candy_LR.hypothesis(data_features)
    print(candy_LR.theta_vector)

print(hypothesis)
print(data_labels)
print(candy_LR.binomial_hypothesis_quantization(hypothesis.numpy()))
print(candy_LR.accuracy(candy_LR.binomial_hypothesis_quantization(hypothesis.numpy()), data_labels.numpy()))
print(candy_LR.hypothesis([torch.tensor([0,0],dtype=torch.float)]))