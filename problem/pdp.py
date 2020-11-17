#from torch.utils.data import Dataset
from torch_geometric.data import Dataset, Data
from torch_geometric.utils import dense_to_sparse
import torch
import os
import pickle
from problem.StatePDP import StatePDP
from utils.beam_search import beam_search  # Should this be here?

HIGH_NUMBER = 10000000

class PDP(object):

    NAME = 'pdp'  # Capacitated Vehicle Routing Problem

    VEHICLE_CAPACITY = 1.0  # (w.l.o.g. vehicle capacity is 1, demands should be scaled)

    @staticmethod
    def get_costs(dataset, pi):
        batch_size, graph_size = dataset['demand'].size()
        # Check that tours are valid, i.e. contain 0 to n -1
        sorted_pi = pi.data.sort(1)[0]

        # Sorting it should give all zeros at front and then 1...n
        assert (
            torch.arange(1, graph_size + 1, out=pi.data.new()).view(1, -1).expand(batch_size, graph_size) ==
            sorted_pi[:, -graph_size:]
        ).all() and (sorted_pi[:, :-graph_size] == 0).all(), "Invalid tour"

        # Visiting depot resets capacity so we add demand = -capacity (we make sure it does not become negative)
        demand_with_depot = torch.cat(
            (
                torch.full_like(dataset['demand'][:, :1], -PDP.VEHICLE_CAPACITY),
                dataset['demand']
            ),
            1
        )
        d = demand_with_depot.gather(1, pi)

        used_cap = torch.zeros_like(dataset['demand'][:, 0])
        for i in range(pi.size(1)):
            used_cap += d[:, i]  # This will reset/make capacity negative if i == 0, e.g. depot visited
            # Cannot use less than 0
            used_cap[used_cap < 0] = 0
            assert (used_cap <= PDP.VEHICLE_CAPACITY + 1e-5).all(), "Used more than capacity"

        # Gather dataset in order of tour
        loc_with_depot = torch.cat((dataset['depot'][:, None, :], dataset['loc']), 1)
        d = loc_with_depot.gather(1, pi[..., None].expand(*pi.size(), loc_with_depot.size(-1)))

        # Length is distance (L2-norm of difference) of each next location to its prev and of first and last to depot
        return (
            (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1)
            + (d[:, 0] - dataset['depot']).norm(p=2, dim=1)  # Depot to first
            + (d[:, -1] - dataset['depot']).norm(p=2, dim=1)  # Last to depot, will be 0 if depot is last
        ), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return PDPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StatePDP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):

        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = PDP.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)


def make_instance(args):
    depot, loc, demand, capacity, *args = args
    grid_size = 1
    if len(args) > 0:
        depot_types, customer_types, grid_size = args
    return {
        'loc': torch.tensor(loc, dtype=torch.float) / grid_size,
        'demand': torch.tensor(demand, dtype=torch.float) / capacity,
        'depot': torch.tensor(depot, dtype=torch.float) / grid_size
    }

class PDPDataset(Dataset):

    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None):
        super(PDPDataset, self).__init__()

        assert size%2 == 0
        self.graph_size = size
        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
            self.data = [make_instance(args) for args in data[offset:offset + num_samples]]

        else:

            # From VRP with RL paper https://arxiv.org/abs/1802.04240
            CAPACITIES = {
                10: 10., #20., I divided by 2 from original because there are fewer calls that is picked up.
                20: 15., #30.
                50: 20., #40.
                100: 25. #50.
            }
            self.data = []
            for i in range(num_samples):
                demand = ((torch.FloatTensor(size // 2).uniform_(0, 9).int() + 1).float() / CAPACITIES[size]).repeat_interleave(2)[:,None]
                demand[1::2] *= -1
                loc = torch.FloatTensor(size, 2).uniform_(0, 1)
                depot = torch.FloatTensor(1, 2).uniform_(0, 1)
                x = torch.cat((loc, depot), dim=0)
                self.data += [
                    Data(
                        x=x,
                        loc=loc,
                        # Uniform 1 - 9, scaled by capacities
                        demand=demand,
                        depot=depot,
                        p_or_d=torch.FloatTensor([((not j%2)*-HIGH_NUMBER + (j%2)*(j-1), (not (j+1)%2)*-HIGH_NUMBER + ((j+1)%2)*(j+1)) for j in range(size)]).int(),
                        edge_index=self._graph_construct()[0]
                    )
                ]
        self.size = len(self.data)
    def _graph_construct(self):
        edges = dense_to_sparse(torch.ones([self.graph_size+1, self.graph_size+1]))
        return edges
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == '__main__':
    a = PDPDataset()
    print(a.data)