import torch
import numpy
from torch_mlir.fx import export_and_import
from torch.export import Dim
from pathlib import Path


class Conv2dWithPaddingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.conv = torch.nn.Conv2d(2, 10, 3, bias=False, padding=3)
        self.train(False)

    def forward(self, x):
        return self.conv(x)


def save_static_conv_module():
    """A basic example of importing a torch.nn.Module to torch-mlir with static dims."""
    sample_shape = (5, 2, 10, 20)
    sample_input = torch.ones(sample_shape)
    torch_module = Conv2dWithPaddingModule()
    test_name = "conv_2d_with_padding_static"

    # save the mlir file
    m_static = export_and_import(torch_module, sample_input, output_type="torch")
    static_path = Path(__file__).parents[1] / "mlir" / f"{test_name}.mlir"
    with open(static_path, "w") as f:
        f.write(m_static.operation.get_asm())

    # save a random input to use
    input_path = Path(__file__).parents[1] / "artifacts" / "inputs" / f"{test_name}.npy"
    random_input = numpy.random.randn(*sample_shape).astype(numpy.float32)
    numpy.save(input_path, random_input)

    # save the output generated by torch
    output_path = (
        Path(__file__).parents[1] / "artifacts" / "outputs" / f"{test_name}_torch.npy"
    )
    torch_output = (
        torch_module(torch.from_numpy(random_input).to(dtype=torch.float32))
        .detach()
        .numpy()
    )
    numpy.save(output_path, torch_output)


def save_dynamic_conv_module():
    """A basic example of importing a torch.nn.Module to torch-mlir with dynamic dims."""
    sample_shape = (5, 2, 10, 20)
    sample_input = torch.ones(sample_shape)
    torch_module = Conv2dWithPaddingModule()
    test_name = "conv_2d_with_padding_dynamic"

    batch = Dim("batch", max=10)
    height = Dim("height", min=7)
    width = Dim("width", min=1)
    dynamic_shapes = {"x": {0: batch, 2: height, 3: width}}

    # save the mlir file
    m_dynamic = export_and_import(
        Conv2dWithPaddingModule(),
        sample_input,
        output_type="torch",
        dynamic_shapes=dynamic_shapes,
    )
    dynamic_path = Path(__file__).parents[1] / "mlir" / f"{test_name}.mlir"
    with open(dynamic_path, "w") as f:
        f.write(m_dynamic.operation.get_asm())

    # save a random input to use
    input_path = Path(__file__).parents[1] / "artifacts" / "inputs" / f"{test_name}.npy"
    random_input = numpy.random.randn(*sample_shape).astype(numpy.float32)
    numpy.save(input_path, random_input)

    # save the output generated by torch
    output_path = (
        Path(__file__).parents[1] / "artifacts" / "outputs" / f"{test_name}_torch.npy"
    )
    torch_output = (
        torch_module(torch.from_numpy(random_input).to(dtype=torch.float32))
        .detach()
        .numpy()
    )
    numpy.save(output_path, torch_output)


def main():
    parent_dir = Path(__file__).parents[1]
    (parent_dir / "mlir").mkdir(parents=True, exist_ok=True)
    (parent_dir / "artifacts" / "inputs").mkdir(parents=True, exist_ok=True)
    (parent_dir / "artifacts" / "outputs").mkdir(parents=True, exist_ok=True)
    (parent_dir / "artifacts" / "vmfb").mkdir(parents=True, exist_ok=True)
    save_static_conv_module()
    save_dynamic_conv_module()


if __name__ == "__main__":
    main()
