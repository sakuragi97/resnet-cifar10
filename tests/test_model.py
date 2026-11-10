import pytest
import torch
from torch import nn

from resnet.config import parse_config
from resnet.engine import evaluate, train_one_epoch
from resnet.model import BasicBlock, ResNet
from resnet.utils import count_parameters, load_checkpoint, save_checkpoint


def test_stage_shapes():
    model = ResNet(depth=20)
    x = torch.randn(2, 3, 32, 32)
    expected = {"stem": (16, 32, 32), "stage1": (16, 32, 32), "stage2": (32, 16, 16), "stage3": (64, 8, 8)}
    for name, shape in expected.items():
        x = getattr(model, name)(x)
        assert tuple(x.shape[1:]) == shape, name
    assert model.head(x).shape == (2, 10)


@pytest.mark.parametrize(
    ("depth", "params"),
    # The paper reports 0.27M, 0.46M, 0.66M, 0.85M and 1.7M parameters.
    [(20, 272_474), (32, 466_906), (44, 661_338), (56, 855_770), (110, 1_730_714)],
)
def test_parameter_count_matches_paper(depth, params):
    assert count_parameters(ResNet(depth)) == params


def test_depth_counts_weighted_layers():
    model = ResNet(depth=32)
    layers = sum(
        isinstance(m, (nn.Conv2d, nn.Linear)) and getattr(m, "kernel_size", (3, 3)) == (3, 3)
        for m in model.modules()
    )
    assert layers == 32  # 3x3 convs + the final linear layer; 1x1 shortcut projections do not count


@pytest.mark.parametrize("depth", [0, 19, 21])
def test_invalid_depth(depth):
    with pytest.raises(ValueError):
        ResNet(depth)


def test_block_with_zero_residual_is_identity():
    block = BasicBlock(16, 16)
    nn.init.zeros_(block.bn2.weight)  # F(x) = 0
    block.eval()
    x = torch.randn(4, 16, 8, 8)
    assert torch.allclose(block(x), torch.relu(x))


def test_projection_shortcut_only_when_shape_changes():
    assert isinstance(BasicBlock(16, 16).downsample, nn.Identity)
    assert isinstance(BasicBlock(16, 32, stride=2).downsample, nn.Sequential)
    assert BasicBlock(16, 32, stride=2, residual=False).downsample is None


def test_plain_network_has_no_shortcut_parameters():
    plain, residual = ResNet(20, residual=False), ResNet(20)
    # The only difference is the two 1x1 conv + BN projections at the start of stage2 and stage3.
    projections = (16 * 32 + 2 * 32) + (32 * 64 + 2 * 64)
    assert count_parameters(residual) - count_parameters(plain) == projections
    assert plain(torch.randn(2, 3, 32, 32)).shape == (2, 10)


def test_zero_init_residual():
    model = ResNet(20, zero_init_residual=True)
    assert all(torch.all(m.bn2.weight == 0) for m in model.modules() if isinstance(m, BasicBlock))


def test_checkpoint_roundtrip(tmp_path):
    model = ResNet(32).eval()
    save_checkpoint(tmp_path / "model.pt", model, {"depth": 32, "residual": True})
    torch.save(model.state_dict(), tmp_path / "bare.pt")
    x = torch.randn(2, 3, 32, 32)
    for name in ["model.pt", "bare.pt"]:
        loaded, _ = load_checkpoint(tmp_path / name)
        assert loaded.depth == 32
        assert torch.allclose(loaded(x), model(x))


def test_training_reduces_loss_on_a_small_batch():
    torch.manual_seed(0)
    images, targets = torch.randn(32, 3, 32, 32), torch.randint(0, 10, (32,))
    loader = [(images, targets)]
    model = ResNet(8)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    device = torch.device("cpu")

    first_loss, _ = evaluate(model, loader, device)
    for _ in range(40):
        train_one_epoch(model, loader, optimizer, device)
    last_loss, accuracy = evaluate(model, loader, device)
    assert last_loss < first_loss / 2
    assert accuracy > 0.5


def test_config_yaml_and_cli_overrides(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("name: test\ndepth: 56\nresidual: true\nepochs: 5\n")
    config = parse_config(["--config", str(path), "--epochs", "7", "--no-residual", "--milestones", "3", "5"])
    assert (config.name, config.depth, config.epochs, config.residual) == ("test", 56, 7, False)
    assert config.milestones == [3, 5]

    path.write_text("dpeth: 56\n")  # typo in a key
    with pytest.raises(TypeError):
        parse_config(["--config", str(path)])
