import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import torch
from fl_core.model_def import get_model
from clients.data_utils import get_dataloader
from clients.training import train_local

SERVER_URL = "http://127.0.0.1:9000"


def _serialize_state(state_dict):
    return {k: v.cpu().tolist() for k, v in state_dict.items()}

# Compute the update (local - global), clip its global L2 norm, and add Gaussian noise if DP enabled.
def _add_dp_to_update(global_state, local_state, clip_norm=1.0, noise_multiplier=0.0, dp_enabled=False):
    update = {k: (local_state[k] - global_state[k]).float() for k in local_state.keys()}

    if not dp_enabled:
        return _serialize_state({k: v for k, v in update.items()})

    # Flatten and compute global L2 norm
    squared = 0.0
    for v in update.values():
        squared += float((v.double() ** 2).sum().item())
    global_norm = squared ** 0.5

    # clip if necessary
    if global_norm > clip_norm and global_norm > 0:
        clip_coef = clip_norm / global_norm
    else:
        clip_coef = 1.0

    clipped_update = {k: (v * clip_coef) for k, v in update.items()}

    # add Gaussian noise to each parameter tensor
    stddev = noise_multiplier * clip_norm
    noised_update = {}
    for k, v in clipped_update.items():
        if stddev > 0:
            noise = torch.normal(mean=0.0, std=stddev, size=v.size(), device=v.device, dtype=v.dtype)
            noised = v + noise
        else:
            noised = v
        noised_update[k] = noised

    return _serialize_state(noised_update)


def client_loop(client_id, dp_enabled=False, noise_multiplier=0.0, clip_norm=1.0, debug=False):
    train_loader, _ = get_dataloader()

    while True:
        print(f"[Client {client_id}] Fetching global model...")
        r = requests.get(f"{SERVER_URL}/get_model").json()
        global_round = r["round"]

        # Load server weights
        model = get_model()
        state = {k: torch.tensor(v) for k, v in r["weights"].items()}
        model.load_state_dict(state)

        print(f"[Client {client_id}] Training locally...")
        model = train_local(model, train_loader, epochs=1)

        # Prepare update: either raw weights or DP-processed (delta, clip, noise)
        # We compute update = local - global and send that (so server can aggregate deltas)
        global_state = {k: v.clone().float() for k, v in state.items()}
        local_state = {k: v.clone().float() for k, v in model.state_dict().items()}

        # If debug and dp enabled, compute raw stats
        raw_sample = None
        raw_norm = None
        if dp_enabled and debug:
            try:
                raw_update = {k: (local_state[k] - global_state[k]).float() for k in global_state.keys()}
                raw_flat = torch.cat([v.view(-1) for v in raw_update.values()])
                raw_sample = raw_flat[:5].cpu().tolist()
                raw_norm = float(raw_flat.norm().item())
            except Exception:
                raw_sample, raw_norm = None, None

        weights_to_send = _add_dp_to_update(global_state, local_state,
                                           clip_norm=clip_norm,
                                           noise_multiplier=noise_multiplier,
                                           dp_enabled=dp_enabled)

        # If debug and dp enabled, compute post stats and print both in presentation-friendly format
        if dp_enabled and debug:
            try:
                # compute global_norm and clip coef similar to _add_dp_to_update
                squared = 0.0
                for v in raw_update.values():
                    squared += float((v.double() ** 2).sum().item())
                global_norm = squared ** 0.5
                clip_coef = (clip_norm / global_norm) if (global_norm > 0 and global_norm > clip_norm) else 1.0
                stddev = noise_multiplier * clip_norm

                # per-tensor pre/post stats for up to 3 keys
                keys = list(raw_update.keys())[:3]
                per_tensor_stats = []
                for k in keys:
                    pre_t = raw_update[k].view(-1)
                    pre_sample = pre_t[:5].cpu().tolist()
                    pre_norm = float(pre_t.norm().item())
                    post_t = torch.tensor(weights_to_send[k]).view(-1)
                    post_sample = post_t[:5].cpu().tolist()
                    post_norm = float(post_t.norm().item())
                    per_tensor_stats.append((k, pre_sample, pre_norm, post_sample, post_norm))

                # Compact table-style debug output for presentation
                header = f"{'key':25} {'#params':>8} {'PRE_norm':>10} {'EXP_noise':>12} {'POST_norm':>12}"
                print('\n--- CLIENT DP DEBUG ---')
                print(f'global_flat_norm: {global_norm:.6f} | clip_norm: {clip_norm} | clip_coef: {clip_coef:.6f} | noise_std: {stddev}')
                print(header)
                print('-' * len(header))
                total_params = 0
                total_expected_noise = 0.0
                for (k, pre_s, pre_n, post_s, post_n) in per_tensor_stats:
                    numel = int(raw_update[k].numel())
                    expected_noise = stddev * (numel ** 0.5)
                    total_params += numel
                    total_expected_noise += expected_noise
                    print(f"{k:25} {numel:8d} {pre_n:10.4f} {expected_noise:12.4f} {post_n:12.4f}")

                print(f"Total params shown: {total_params} | Total expected_noise (approx): {total_expected_noise:.4f}")
                print('--- END DEBUG ---\n')
            except Exception as e:
                print(f"[Client DEBUG] failed to compute detailed stats: {e}")

        print(f"[Client {client_id}] Sending update... (DP={'ON' if dp_enabled else 'OFF'})")
        requests.post(f"{SERVER_URL}/submit_update", json={
            "weights": weights_to_send,
            "data_size": len(train_loader.dataset)
        })

        print(f"[Client {client_id}] Update sent. Waiting next round...\n")
        import time
        time.sleep(3)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Federated client with optional DP on updates")
    parser.add_argument("client_id", nargs="?", default="1", help="Client id string")
    parser.add_argument("--dp", action="store_true", help="Enable differential privacy (clip + noise)")
    parser.add_argument("--noise", type=float, default=0.0, help="Noise multiplier (std = noise * clip)")
    parser.add_argument("--clip", type=float, default=1.0, help="Clip norm for DP (L2)" )
    parser.add_argument("--debug", action="store_true", help="Enable client-side debug prints (works with --dp)")

    args = parser.parse_args()
    # pass debug flag through to client loop; debug prints are gated by dp_enabled && args.debug
    client_loop(args.client_id, dp_enabled=args.dp, noise_multiplier=args.noise, clip_norm=args.clip, debug=args.debug)
