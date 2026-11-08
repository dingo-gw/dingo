import pytest
import torch
import torch.nn as nn

from dingo.core.utils import torchutils


def test_get_activation_function_from_string():
    activation = torchutils.get_activation_function_from_string("elu")
    # Returns a callable activation (torch.nn.functional.elu).
    assert callable(activation)
    out = activation(torch.tensor([-1.0, 1.0]))
    assert out.shape == (2,)


def test_get_activation_function_unknown_raises():
    with pytest.raises(ValueError):
        torchutils.get_activation_function_from_string("not_an_activation")


def test_get_optimizer_from_kwargs():
    model = nn.Linear(2, 2)
    optimizer = torchutils.get_optimizer_from_kwargs(
        model.parameters(), type="adam", lr=0.01
    )
    assert isinstance(optimizer, torch.optim.Adam)


def test_get_scheduler_from_kwargs():
    model = nn.Linear(2, 2)
    optimizer = torchutils.get_optimizer_from_kwargs(
        model.parameters(), type="adam", lr=0.01
    )
    scheduler = torchutils.get_scheduler_from_kwargs(optimizer, type="cosine", T_max=10)
    assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)


def test_split_dataset_into_train_and_test():
    dataset = torch.utils.data.TensorDataset(torch.arange(100.0))
    train, test = torchutils.split_dataset_into_train_and_test(
        dataset, 0.8, block_size=1
    )
    assert len(train) == 80
    assert len(test) == 20
    # The split is a partition (no shared indices, full coverage).
    assert len(train) + len(test) == len(dataset)


@pytest.mark.parametrize("n, fraction", [(0, 0.8), (7, 0.5), (1000, 0.8), (1001, 0.95)])
def test_split_block_size_1_is_random_split(n, fraction):
    """Runs started before block_size existed resume with the same train/test split."""
    dataset = torch.utils.data.TensorDataset(torch.arange(float(n)))
    train, test = torchutils.split_dataset_into_train_and_test(
        dataset, fraction, block_size=1
    )
    size = int(fraction * n)
    ref_train, ref_test = torch.utils.data.random_split(
        dataset, [size, n - size], generator=torch.Generator().manual_seed(42)
    )
    assert train.indices == ref_train.indices and test.indices == ref_test.indices


@pytest.mark.parametrize(
    "block_size, test_rows",
    [(1, [673, 842, 746, 582, 717, 364]), (16, [332, 333, 334, 335, 336, 337])],
)
def test_split_is_frozen(block_size, test_rows):
    """The split of saved models must never change (also not with torch's randperm):
    a resumed run would train on its test rows."""
    dataset = torch.utils.data.TensorDataset(torch.arange(1000))
    _, test = torchutils.split_dataset_into_train_and_test(
        dataset, 0.8, block_size=block_size
    )
    assert test.indices[:6] == test_rows


def num_runs(rows):
    """Number of runs of consecutive rows in a sequence."""
    rows = torch.as_tensor(rows)
    return 1 + int((rows[1:] != rows[:-1] + 1).sum()) if len(rows) else 0


def test_block_size_split_and_sampler():
    """Blocks of the split and the samplers: a partition into whole blocks, every
    training row once per epoch, batches of few contiguous runs, a reproducible order
    per epoch, and contiguous, disjoint DDP shares."""
    n, b, B = 1000, 16, 64
    dataset = torch.utils.data.TensorDataset(torch.arange(n))
    train_loader, test_loader = torchutils.build_train_and_test_loaders(
        dataset, 0.8, B, num_workers=0, block_size=b
    )
    sampler = train_loader.sampler
    train, test = train_loader.dataset.indices, test_loader.dataset.indices
    assert sorted(train + test) == list(range(n))
    assert num_runs(test) <= len(test) / b + 2

    orders = []
    for epoch in (1, 1, 2):
        sampler.set_epoch(epoch)
        batches = [x.tolist() for (x,) in train_loader]
        assert sorted(sum(batches, [])) == sorted(train)
        assert all(num_runs(x) <= 2 * (B / b + 2) + 1 for x in batches)
        orders.append(sum(batches, []))
    assert orders[0] == orders[1] and orders[0] != orders[2]
    assert num_runs(sum([x.tolist() for (x,) in test_loader], [])) <= len(test) / b + 2

    W = 3
    shares = []
    for rank in range(W):
        loader, _ = torchutils.build_train_and_test_loaders(
            dataset, 0.8, B, num_workers=0, world_size=W, rank=rank, block_size=b
        )
        loader.sampler.set_epoch(1)
        batches = [x.tolist() for (x,) in loader]
        assert all(num_runs(x) <= 2 * (B / b + 2) + 1 for x in batches)
        shares.append(set(sum(batches, [])))
    assert {len(x) for x in shares} == {len(train) // W}
    assert set.union(*shares) <= set(train) and len(set.union(*shares)) > len(train) - W
    for bad in (
        {"block_size": 64.0},
        {"block_size": True},
        {"num_replicas": 2, "rank": 2},
    ):
        with pytest.raises(ValueError):
            torchutils.BlockSampler(n, **bad)

    # The block grid's phase (the random offset) changes between epochs.
    s = torchutils.BlockSampler(1024, b)
    phases = set()
    for epoch in range(8):
        s.set_epoch(epoch)
        phases.add(next(iter(s)) % b)
    assert len(phases) > 1


def test_get_number_of_model_parameters():
    model = nn.Linear(2, 3)  # weight 2*3 + bias 3 = 9
    assert torchutils.get_number_of_model_parameters(model) == 9


def test_get_lr_returns_optimizer_learning_rates():
    optimizer = torchutils.get_optimizer_from_kwargs(
        nn.Linear(2, 2).parameters(), type="adam", lr=0.007
    )
    assert torchutils.get_lr(optimizer) == [0.007]


def test_torch_detach_to_cpu():
    tensor = torch.ones(3, requires_grad=True)
    detached = torchutils.torch_detach_to_cpu(tensor)
    assert not detached.requires_grad
    # Non-tensor inputs pass through unchanged.
    assert torchutils.torch_detach_to_cpu(5) == 5


def test_set_requires_grad_flag_by_name():
    model = nn.Linear(2, 3)
    torchutils.set_requires_grad_flag(
        model, name_contains="weight", requires_grad=False
    )
    assert model.weight.requires_grad is False
    assert model.bias.requires_grad is True
