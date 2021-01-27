from problem.pdp import PDP
import os
import pickle
from torch import manual_seed


def check_extension(filename):
    if os.path.splitext(filename)[1] != ".pkl":
        return filename + ".pkl"
    return filename


def save_dataset(dataset, filename):

    filedir = os.path.split(filename)[0]

    if not os.path.isdir(filedir):
        os.makedirs(filedir)

    with open(check_extension(filename), 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)


def load_dataset(filename):

    with open(check_extension(filename), 'rb') as f:
        return pickle.load(f)


GRAPH_SIZE = 20
NUM_SAMPLES = 1000
NAME = "TEST2"
SEED = 1235

manual_seed(SEED)
dataset = PDP.make_dataset(size=GRAPH_SIZE, num_samples=NUM_SAMPLES)
save_dataset(dataset, f'data/pdp/pdp{GRAPH_SIZE}_{NAME}_seed{SEED}.pkl')



