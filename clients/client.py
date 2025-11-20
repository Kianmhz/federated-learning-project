import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import requests
import torch

from clients.data_utils import get_client_loaders, get_dataloader
from clients.training import train_local
from fl_core.model_def import get_model

SERVER_URL = "http://127.0.0.1:9000"


def _serialize_state(state_dict):
    return {k: v.cpu().tolist() for k, v in state_dict.items()}


# Compute the update (local - global), clip its global L2 norm, and add Gaussian noise if DP enabled.
def _add_dp_to_update(
    global_state, local_state, clip_norm=1.0, noise_multiplier=0.0, dp_enabled=False
):
    update = {k: (local_state[k] - global_state[k]).float() for k in local_state.keys()}

    if not dp_enabled:
        return _serialize_state({k: v for k, v in update.items()})

    # Flatten and compute global L2 norm
    squared = 0.0
    for v in update.values():
        squared += float((v.double() ** 2).sum().item())
    global_norm = squared**0.5

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
            noise = torch.normal(
                mean=0.0, std=stddev, size=v.size(), device=v.device, dtype=v.dtype
            )
            noised = v + noise
        else:
            noised = v
        noised_update[k] = noised

    return _serialize_state(noised_update)


def run_client_loop(
    train_loader,
    client_id,
    dp_enabled=False,
    noise_multiplier=0.0,
    clip_norm=1.0,
    debug=False,
):
    """Run the main client loop using the provided train_loader."""
    while True:
        print(f"[Client {client_id}] Fetching global model...")
        r = requests.get(f"{SERVER_URL}/get_model").json()

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
        if dp_enabled and debug:
            raw_update = {
                k: (local_state[k] - global_state[k]).float()
                for k in global_state.keys()
            }

        weights_to_send = _add_dp_to_update(
            global_state,
            local_state,
            clip_norm=clip_norm,
            noise_multiplier=noise_multiplier,
            dp_enabled=dp_enabled,
        )

        # If debug and dp enabled, compute post stats and print both
        if dp_enabled and debug:
            try:
                # compute global_norm and clip coef similar to _add_dp_to_update
                squared = 0.0
                for v in raw_update.values():
                    squared += float((v.double() ** 2).sum().item())
                global_norm = squared**0.5
                clip_coef = (
                    (clip_norm / global_norm)
                    if (global_norm > 0 and global_norm > clip_norm)
                    else 1.0
                )
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
                    per_tensor_stats.append(
                        (k, pre_sample, pre_norm, post_sample, post_norm)
                    )

                # Compact table-style debug output
                header = f"{'key':25} {'#params':>8} {'PRE_norm':>10} {'EXP_noise':>12} {'POST_norm':>12}"
                print("\n--- CLIENT DP DEBUG ---")
                print(
                    f"global_flat_norm: {global_norm:.6f} | clip_norm: {clip_norm} | clip_coef: {clip_coef:.6f} | noise_std: {stddev}"
                )
                print(header)
                print("-" * len(header))
                total_params = 0
                total_expected_noise = 0.0
                for k, _, pre_n, _, post_n in per_tensor_stats:
                    numel = int(raw_update[k].numel())
                    expected_noise = stddev * (numel**0.5)
                    total_params += numel
                    total_expected_noise += expected_noise
                    print(
                        f"{k:25} {numel:8d} {pre_n:10.4f} {expected_noise:12.4f} {post_n:12.4f}"
                    )

                print(
                    f"Total params shown: {total_params} | Total expected_noise (approx): {total_expected_noise:.4f}"
                )
                print("--- END DEBUG ---\n")
            except Exception as e:
                print(f"[Client DEBUG] failed to compute detailed stats: {e}")

        print(
            f"[Client {client_id}] Sending update... (DP={'ON' if dp_enabled else 'OFF'})"
        )
        requests.post(
            f"{SERVER_URL}/submit_update",
            json={
                "client_id": str(
                    client_id
                ),  # added id so that server can identify client
                "weights": weights_to_send,
                "data_size": len(train_loader.dataset),
            },
        )

        print(f"[Client {client_id}] Update sent. Waiting next round...\n")
        import time

        time.sleep(3)


def client_loop(
    client_id, dp_enabled=False, noise_multiplier=0.0, clip_norm=1.0, debug=False
):
    # default: load full dataset (backwards-compatible). If caller set CLIENT_LOADERS
    # externally they can pass a per-client loader via closure; here we default to full loader.
    train_loader, _ = get_dataloader()
    run_client_loop(
        train_loader,
        client_id,
        dp_enabled=dp_enabled,
        noise_multiplier=noise_multiplier,
        clip_norm=clip_norm,
        debug=debug,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Federated client with optional DP on updates"
    )
    parser.add_argument("client_id", nargs="?", default="1", help="Client id string")
    parser.add_argument(
        "--dp", action="store_true", help="Enable differential privacy (clip + noise)"
    )
    parser.add_argument(
        "--noise", type=float, default=0.0, help="Noise multiplier (std = noise * clip)"
    )
    parser.add_argument("--clip", type=float, default=1.0, help="Clip norm for DP (L2)")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable client-side debug prints (works with --dp)",
    )
    # options to control client-side data partitioning
    parser.add_argument(
        "--num-clients",
        type=int,
        default=10,
        help="Total number of clients for splitting train data",
    )
    parser.add_argument(
        "--non-iid",
        action="store_true",
        help="Use Dirichlet non-IID split instead of IID",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Dirichlet alpha parameter (smaller -> more skew)",
    )

    args = parser.parse_args()
    # Presentation-friendly confirmation: print whether we're running IID or non-IID
    if args.non_iid:
        print(
            f"[CLIENT] Running with non-IID Dirichlet split (num_clients={args.num_clients}, alpha={args.alpha})"
        )
    else:
        print(f"[CLIENT] Running with IID split (num_clients={args.num_clients})")

    # If user requested per-client partitioning, create client loaders and pick the client's loader
    if args.num_clients is not None:
        clients_loaders, _ = get_client_loaders(
            num_clients=args.num_clients, iid=(not args.non_iid), alpha=args.alpha
        )

        # replace get_dataloader behavior by injecting local loader into client_loop via closure
        def client_loop_with_local_loader(
            client_id,
            dp_enabled=False,
            noise_multiplier=0.0,
            clip_norm=1.0,
            debug=False,
        ):
            try:
                cid = int(client_id)
            except Exception:
                cid = 0
            # pick loader for this client id
            train_loader = clients_loaders.get(cid, None)
            # Validate requested client id
            if cid < 0 or cid >= args.num_clients:
                print(
                    f"[CLIENT] ERROR: client_id={cid} out of range for num_clients={args.num_clients}. Falling back to full loader."
                )
                train_loader, _ = get_dataloader()
            else:
                # report selected partition size
                if train_loader is not None:
                    try:
                        print(
                            f"[CLIENT] Loaded partition for client {cid}: {len(train_loader.dataset)} samples"
                        )
                    except Exception:
                        print(f"[CLIENT] Loaded partition for client {cid}")
            if train_loader is None:
                # fallback to full loader
                print(
                    f"[Client {client_id}] WARNING: no local loader found, using full dataset loader"
                )
                train_loader, _ = get_dataloader()

            # use the shared run_client_loop to avoid duplication
            run_client_loop(
                train_loader,
                client_id,
                dp_enabled=dp_enabled,
                noise_multiplier=noise_multiplier,
                clip_norm=clip_norm,
                debug=debug,
            )

        # run the modified client loop
        client_loop_with_local_loader(
            args.client_id,
            dp_enabled=args.dp,
            noise_multiplier=args.noise,
            clip_norm=args.clip,
            debug=args.debug,
        )
    else:
        # pass debug flag through to client loop; debug prints are gated by dp_enabled && args.debug
        client_loop(
            args.client_id,
            dp_enabled=args.dp,
            noise_multiplier=args.noise,
            clip_norm=args.clip,
            debug=args.debug,
        )
