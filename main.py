import argparse
import random

import torch
import torch.nn as nn
import torch.optim as optim

from model import GGNN
from utils.train import train
from utils.test import test
from utils.data.dataset import bAbIDataset
from utils.data.dataloader import bAbIDataloader

parser = argparse.ArgumentParser()
parser.add_argument('--task_id', type=int, default=4, help='bAbI task id')
parser.add_argument('--question_id', type=int, default=0, help='question types')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=10, help='input batch size')
parser.add_argument('--state_dim', type=int, default=4, help='GGNN hidden state size')
parser.add_argument('--n_steps', type=int, default=5, help='propogation steps number of GGNN')
parser.add_argument('--niter', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--cuda', action='store_true', default=False, help='enables cuda')
parser.add_argument('--verbal', action='store_true', help='print training info or not')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

opt.dataroot = 'babi_data/processed_1/train/%d_graphs.txt' % opt.task_id

if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

# Talking the example of task 4:
#
# 1 1 2
# 2 1 3
# ? 1 2 1
#
# The first two rows are edge_list.
# The final row is target_list.
#
# Name for every number
# node_id   edge    node_id
# node_id   edge    node_id
# ? task_id annotation  task_output
#
# task_id is task(question) type. You can consider it is edge.
# node_id is node type
# annotation is one-hot encoding from number in target_list[1], such as: 2 become [0, 1, 0, 0]

# annotation can be considered as (left) node and task_id can be considered as edge.
# task_output can be considered as (right) node. But the edge (task_id) use different encoding.
# detail is below:

# edge_type.txt:
# e=2
# n=1
# e is east. n is north.

# question_type.txt:
# s=1
# e=2
# w=3
# n=4
# s is south, e is east, w is west, n is north.

# So:
# 1 1 2
# 2 1 3
# ? 1 2 1
# ->
# 1 n 2     [n come from edge_type.txt]
# 2 n 3     [n come from edge_type.txt]
# ? s 2 1   [s come from question_type.txt]
# ->
# -> 2 should be in front of direction.
# We can infer the answer 1 in "2 s 1" from "1 n 2" and "2 n 3"
# So the final answer is:
# 1 n 2
# 2 n 3
# 2 s 1
#
# One more example:
# 2 2 3
# 3 2 1
# ? 2 3 1
# ->
# 2 e 3     [e come from edge_type.txt]
# 3 e 1     [e come from edge_type.txt]
# ? e 3 1   [e come from question_type.txt]
# ->
# -> 3 should be in front of direction.
# We can infer the answer 1 in "3 e 1" from "2 e 3" and "3 e 1"
# So the final answer is:
# 2 e 3
# 3 e 1
# 3 e 1
# Actually the question_type is also a edge, but it come from question_type.txt not edge_type.txt.
# So the question here is predict (reason) the node if we know its edge and its node.
# The easy point is we separate four kinds of question, which means we do not need to learn the edge in the question.

# ----------------------------------------------

# Talking the example of task 4:
# But let us see a more difficult task which is 19:
# Here only have one question type, and the output is the path.
# Example:
# 1 The office is east of the hallway.
# 2 The kitchen is north of the office.
# 3 The garden is west of the bedroom.
# 4 The office is west of the garden.
# 5 The bathroom is north of the garden.
# 6 How do you go from the kitchen to the garden?
# Pathe: south,east. Find Path from sentence:2 4
# Data in here:
# 1 1 2
# 1 2 3
# 4 2 1
# 5 2 4
# ? 1 5 1 1 1
# edge_type.txt:
# s=3
# e=1
# w=4
# n=2
# question_type.txt:
# path=1

# So, the question become:
# 1 e 2
# 1 n 3
# 4 n 1
# 5 n 4
# ? path 5 1 1 1
# Adjust the order by referring to the previous method, the question is:
# question: 5 path 1, answer: 1 1
# Back to the direction, we need to refer the 19_labels.txt:
# s=4
# e=3
# w=2
# n=1
# So 1 is n, we can conclude:
# 1 e 2
# 1 n 3
# 4 n 1
# 5 n 4
# 5 path 1 -> n n
# (The number of edge label(edge_type.txt) is different from labels.txt)

# Fortunately, all mission can directly get the answer from the training data and you don't need to infer.
# For example, there is no such example:
# 1 e 2
# 1 n 3
# 1 n 4 [I change here from before]
# 5 n 4
# 5 path 1 -> n s
# In this example, you need to infer more information because the path also start from left to right.
# But I cannot find these example in the dataset. So the inference here is relatively simple.

# More easily, the model only predict the destination not the path, which is :
# Input:
# 1 e 2
# 1 n 3
# 4 n 1
# 5 n 4
# 5 path,(path is useless, so it will not be a input)
# Output:
# 1
# Remove:
# n n (1 1)

# I think it is easy to find the destination, because all path is 2, so you can find the destination easily.

def main(opt):
    train_dataset = bAbIDataset(opt.dataroot, opt.question_id, True)
    train_dataloader = bAbIDataloader(train_dataset, batch_size=opt.batchSize, \
                                      shuffle=True, num_workers=2) # num_workers is working thread
    # Only return a bit of data whose task_id equal to opt.question_id
    # train_dataloader can be splitted into three object: (1) edge_list; (2) annotation; (3) task_output


    test_dataset = bAbIDataset(opt.dataroot, opt.question_id, False)
    test_dataloader = bAbIDataloader(test_dataset, batch_size=opt.batchSize, \
                                     shuffle=False, num_workers=2)

    opt.annotation_dim = 1  # for bAbI
    opt.n_edge_types = train_dataset.n_edge_types
    opt.n_node = train_dataset.n_node

    net = GGNN(opt)
    net.double()
    print(net)

    criterion = nn.CrossEntropyLoss()

    if opt.cuda:
        net.cuda()
        criterion.cuda()

    optimizer = optim.Adam(net.parameters(), lr=opt.lr)

    for epoch in range(0, opt.niter):
        train(epoch, train_dataloader, net, criterion, optimizer, opt)
        test(test_dataloader, net, criterion, optimizer, opt)


if __name__ == "__main__":
    main(opt)

