import torch
from fl_core.model_def import get_model
from copy import deepcopy

class Aggregator:
    def __init__(self):
        self.global_model = get_model()
        self.current_round = 0
        self.updates = []
        self.data_sizes = []

    def get_weights(self):
        return {k: v.cpu().tolist() for k, v in self.global_model.state_dict().items()}

    def receive_update(self, weights, data_size):
        self.updates.append(weights)
        self.data_sizes.append(data_size)

    def aggregate(self):
        if len(self.updates) == 0:
            return
        # We'll treat incoming 'weights' as updates/deltas (local - global).
        # Compute weighted average delta and add it to the current global model.
        global_state = deepcopy(self.global_model.state_dict())

        # Weighted averaging of deltas
        total_data = sum(self.data_sizes)

        avg_delta = {}
        for key in global_state.keys():
            avg_delta_tensor = sum(
                (torch.tensor(update[key]) * (ds / total_data))
                for update, ds in zip(self.updates, self.data_sizes)
            )
            avg_delta[key] = avg_delta_tensor

        # Apply averaged delta: global + avg_delta
        new_state = {k: (global_state[k].float() + avg_delta[k].float()) for k in global_state.keys()}

        self.global_model.load_state_dict(new_state)
        self.current_round += 1
        self.updates = []
        self.data_sizes = []
