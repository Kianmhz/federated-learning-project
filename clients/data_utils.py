import collections
import numpy as np
import torch
from torchvision import datasets, transforms


def _load_mnist(download=True):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train = datasets.MNIST("./data", train=True, download=download, transform=transform)
    test = datasets.MNIST("./data", train=False, download=download, transform=transform)
    return train, test

def get_dataloader(batch_size=32, download=True):
    train, test = _load_mnist(download=download)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# Split dataset indices IID among num_clients as evenly as possible.
def iid_split(dataset, num_clients):
    n = len(dataset)
    all_indices = np.arange(n)
    np.random.shuffle(all_indices)
    splits = np.array_split(all_indices, num_clients)
    return {i: list(s) for i, s in enumerate(splits)}

# Split dataset indices into non-IID client distributions using a Dirichlet allocation over labels.
# alpha: Dirichlet concentration parameter (smaller -> more skew)
# min_size: minimum number of samples per client (retry until satisfied)
def dirichlet_noniid_split(dataset, num_clients, alpha=0.5, min_size=10):
    # get labels as numpy array
    try:
        labels = np.array(dataset.targets)
    except Exception:
        # some dataset variants store targets as list
        labels = np.array(list(dataset.targets))

    num_classes = int(labels.max()) + 1
    indices_by_class = [np.where(labels == y)[0] for y in range(num_classes)]

    while True:
        client_indices = {i: [] for i in range(num_clients)}
        # sample class proportions for each client
        proportions = np.random.dirichlet([alpha] * num_clients, size=num_classes)

        for cls, cls_idx in enumerate(indices_by_class):
            cls_idx = cls_idx.copy()
            np.random.shuffle(cls_idx)
            # split cls_idx according to proportions[cls]
            props = proportions[cls]
            # convert proportions to counts
            counts = (props * len(cls_idx)).astype(int)
            # adjust to match total
            diff = len(cls_idx) - counts.sum()
            if diff > 0:
                for k in range(diff):
                    counts[k % num_clients] += 1
            elif diff < 0:
                for k in range(-diff):
                    counts[k % num_clients] -= 1

            pointer = 0
            for client_id, c in enumerate(counts):
                if c > 0:
                    client_indices[client_id].extend(cls_idx[pointer:pointer + c].tolist())
                    pointer += c

        sizes = [len(v) for v in client_indices.values()]
        if min(sizes) >= min_size:
            break

    return client_indices

# Get per-client DataLoaders for MNIST and a global test loader.
# iid: if True use IID split, else use Dirichlet non-IID with given alpha
# alpha: Dirichlet concentration parameter used when iid=False
def get_client_loaders(num_clients=10, iid=True, alpha=0.5, batch_size=32, download=True):
    train, test = _load_mnist(download=download)

    if iid:
        splits = iid_split(train, num_clients)
    else:
        splits = dirichlet_noniid_split(train, num_clients, alpha=alpha)

    client_loaders = {}
    for cid, idxs in splits.items():
        subset = torch.utils.data.Subset(train, idxs)
        loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=True)
        client_loaders[cid] = loader

    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)
    return client_loaders, test_loader


def _label_distribution(loader):
    counts = collections.Counter()
    for x, y in loader.dataset:
        counts[int(y)] += 1
    return counts


if __name__ == "__main__":
    # smoke test printing sizes and label distributions for IID and non-IID
    import pprint

    print("Preparing IID split (10 clients)...")
    clients_iid, _ = get_client_loaders(num_clients=10, iid=True, batch_size=64)
    sizes_iid = {k: len(v.dataset) for k, v in clients_iid.items()}
    print("Client sizes (IID):")
    pprint.pprint(sizes_iid)

    print("\nPreparing non-IID split (10 clients, alpha=0.1)...")
    clients_noniid, _ = get_client_loaders(num_clients=10, iid=False, alpha=0.1, batch_size=64)
    sizes_noniid = {k: len(v.dataset) for k, v in clients_noniid.items()}
    print("Client sizes (non-IID):")
    pprint.pprint(sizes_noniid)

    # print label distributions for a few clients
    print("\nLabel distribution for client 0 (IID):")
    pprint.pprint(_label_distribution(clients_iid[0]), width=200)
    print("Label distribution for client 0 (non-IID):")
    pprint.pprint(_label_distribution(clients_noniid[0]))