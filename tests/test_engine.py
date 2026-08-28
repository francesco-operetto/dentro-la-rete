"""Numerical cross-check of the application's neural-network core against PyTorch.

The JavaScript fixtures are generated from the actual mathematical-core <script>
inside the application HTML. PyTorch rebuilds the same networks in float64 and
uses autograd to obtain the reference derivatives.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn.functional as F

FIXTURE_PATH = Path(__file__).with_name("fixtures") / "generated_reference.json"
RTOL = 1e-12
ATOL = 1e-13
FD_RTOL = 3e-7
FD_ATOL = 3e-9
FD_H = 1e-6

# JavaScript Number is binary64, so the reference must use the same precision.
torch.set_default_dtype(torch.float64)


def load_fixtures() -> dict[str, Any]:
    if not FIXTURE_PATH.exists():
        pytest.fail(
            f"Missing {FIXTURE_PATH}. Run `node tests/generate_fixtures.mjs` first.",
            pytrace=False,
        )
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


FIXTURES = load_fixtures()


def tensor(values: Any, *, requires_grad: bool = False) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float64, requires_grad=requires_grad)


def assert_tensor_close(actual: torch.Tensor, expected: Any, label: str) -> None:
    expected_tensor = tensor(expected)
    try:
        torch.testing.assert_close(actual.detach(), expected_tensor, rtol=RTOL, atol=ATOL)
    except AssertionError as exc:
        raise AssertionError(f"{label}\n{exc}") from exc


def assert_scalar_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=RTOL, abs_tol=ATOL):
        raise AssertionError(
            f"{label}: actual={actual:.17g}, expected={expected:.17g}, "
            f"abs diff={abs(actual - expected):.3e}"
        )


class TorchReference:
    """Same network topology, with gradients supplied independently by autograd."""

    def __init__(self, fixture: dict[str, Any], parameters: dict[str, Any] | None = None):
        self.arch = fixture["arch"]
        self.activation = fixture["activation"]
        self.output = fixture["output"]
        source = parameters or fixture["initialParameters"]
        self.weights = [tensor(w, requires_grad=True) for w in source["weights"]]
        self.biases = [tensor(b, requires_grad=True) for b in source["bias"]]

    def zero_grad(self) -> None:
        for parameter in [*self.weights, *self.biases]:
            parameter.grad = None

    def hidden_activation(self, z: torch.Tensor) -> torch.Tensor:
        if self.activation == "sigmoid":
            return torch.sigmoid(z)
        if self.activation == "tanh":
            return torch.tanh(z)
        if self.activation == "relu":
            return torch.relu(z)
        if self.activation == "linear":
            return z
        raise ValueError(f"Unknown activation: {self.activation}")

    def forward(self, x_values: list[float]) -> tuple[list[torch.Tensor | None], list[torch.Tensor]]:
        a = tensor(x_values)
        activations: list[torch.Tensor] = [a]
        zs: list[torch.Tensor | None] = [None]
        last_layer = len(self.weights) - 1

        for layer, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            z = F.linear(a, weight, bias)
            if z.requires_grad:
                z.retain_grad()
            zs.append(z)

            if layer == last_layer:
                output_type = self.output["type"]
                if output_type == "multiclass":
                    a = torch.softmax(z, dim=0)
                elif output_type == "regression":
                    a = z
                else:
                    a = torch.sigmoid(z)
            else:
                a = self.hidden_activation(z)
            activations.append(a)

        return zs, activations

    def loss(self, outputs: torch.Tensor, y: int | float) -> torch.Tensor:
        if self.output["type"] == "multiclass":
            # The application's 1e-12 clamp is inactive in these fixtures. Keeping
            # it here matches the documented loss while autograd supplies the gradient.
            return -torch.log(torch.clamp(outputs[int(y)], min=1e-12))
        target = tensor(float(y))
        return 0.5 * (outputs[0] - target) ** 2

    def compute(self, x_values: list[float], y: int | float) -> dict[str, Any]:
        self.zero_grad()
        zs, activations = self.forward(x_values)
        outputs = activations[-1]
        loss = self.loss(outputs, y)
        loss.backward()

        return {
            "zs": zs,
            "activations": activations,
            "outputs": outputs,
            "loss": loss,
            "delta": [None] + [z.grad for z in zs[1:]],
            "gradW": [w.grad for w in self.weights],
            "gradB": [b.grad for b in self.biases],
        }

    def apply_gradient_descent(self, learning_rate: float) -> None:
        with torch.no_grad():
            for weight, bias in zip(self.weights, self.biases):
                weight.add_(weight.grad, alpha=-learning_rate)
                bias.add_(bias.grad, alpha=-learning_rate)

    def parameter_snapshot(self) -> dict[str, list[Any]]:
        return {
            "weights": [w.detach().tolist() for w in self.weights],
            "bias": [b.detach().tolist() for b in self.biases],
        }


def compare_step(reference: TorchReference, result: dict[str, Any], fixture_step: dict[str, Any], learning_rate: float, prefix: str) -> None:
    for layer in range(1, len(result["zs"])):
        assert_tensor_close(result["zs"][layer], fixture_step["zs"][layer], f"{prefix}: z layer {layer}")

    for layer, activation in enumerate(result["activations"]):
        assert_tensor_close(activation, fixture_step["as"][layer], f"{prefix}: activation layer {layer}")

    assert_tensor_close(result["outputs"], fixture_step["outputs"], f"{prefix}: outputs")
    assert_scalar_close(float(result["loss"].detach()), fixture_step["err"], f"{prefix}: loss")

    for layer in range(1, len(result["delta"])):
        assert_tensor_close(result["delta"][layer], fixture_step["delta"][layer], f"{prefix}: delta layer {layer}")

    for layer, grad in enumerate(result["gradW"]):
        assert_tensor_close(grad, fixture_step["gradW"][layer], f"{prefix}: gradW layer {layer}")
        assert_tensor_close(result["gradB"][layer], fixture_step["gradB"][layer], f"{prefix}: gradB layer {layer}")
        assert_tensor_close(-learning_rate * grad, fixture_step["dW"][layer], f"{prefix}: dW layer {layer}")
        assert_tensor_close(-learning_rate * result["gradB"][layer], fixture_step["dB"][layer], f"{prefix}: dB layer {layer}")


def compare_parameters(actual: dict[str, Any], expected: dict[str, Any], prefix: str) -> None:
    for layer, values in enumerate(actual["weights"]):
        assert_tensor_close(tensor(values), expected["weights"][layer], f"{prefix}: weights layer {layer}")
    for layer, values in enumerate(actual["bias"]):
        assert_tensor_close(tensor(values), expected["bias"][layer], f"{prefix}: bias layer {layer}")


def functional_loss(fixture: dict[str, Any], parameters: dict[str, Any], x_values: list[float], y: int | float) -> float:
    reference = TorchReference(fixture, parameters=parameters)
    with torch.no_grad():
        _, activations = reference.forward(x_values)
        return float(reference.loss(activations[-1], y))


def finite_difference(fixture: dict[str, Any], kind: str, layer: int, j: int, i: int | None = None) -> float:
    plus = json.loads(json.dumps(fixture["initialParameters"]))
    minus = json.loads(json.dumps(fixture["initialParameters"]))

    if kind == "weight":
        assert i is not None
        plus["weights"][layer][j][i] += FD_H
        minus["weights"][layer][j][i] -= FD_H
    else:
        plus["bias"][layer][j] += FD_H
        minus["bias"][layer][j] -= FD_H

    loss_plus = functional_loss(fixture, plus, fixture["x"], fixture["y"])
    loss_minus = functional_loss(fixture, minus, fixture["x"], fixture["y"])
    return (loss_plus - loss_minus) / (2 * FD_H)


@pytest.mark.parametrize("fixture", FIXTURES["cases"], ids=lambda item: item["name"])
def test_single_step_against_pytorch_autograd(fixture: dict[str, Any]) -> None:
    reference = TorchReference(fixture)
    result = reference.compute(fixture["x"], fixture["y"])
    compare_step(reference, result, fixture["step"], fixture["learningRate"], fixture["name"])

    reference.apply_gradient_descent(fixture["learningRate"])
    compare_parameters(reference.parameter_snapshot(), fixture["updatedParameters"], fixture["name"])


@pytest.mark.parametrize(
    "fixture",
    [item for item in FIXTURES["cases"] if item.get("finiteDifference", True)],
    ids=lambda item: item["name"],
)
def test_selected_derivatives_against_central_finite_differences(fixture: dict[str, Any]) -> None:
    # A numerical derivative is less precise than autograd, but provides a third,
    # independent check of the sign and magnitude of representative derivatives.
    weight_fd = finite_difference(fixture, "weight", 0, 0, 0)
    weight_grad = fixture["step"]["gradW"][0][0][0]
    assert math.isclose(weight_fd, weight_grad, rel_tol=FD_RTOL, abs_tol=FD_ATOL), (
        f"{fixture['name']}: finite-difference weight derivative {weight_fd:.17g} "
        f"!= JavaScript gradient {weight_grad:.17g}"
    )

    last_layer = len(fixture["arch"]) - 2
    bias_fd = finite_difference(fixture, "bias", last_layer, 0)
    bias_grad = fixture["step"]["gradB"][last_layer][0]
    assert math.isclose(bias_fd, bias_grad, rel_tol=FD_RTOL, abs_tol=FD_ATOL), (
        f"{fixture['name']}: finite-difference bias derivative {bias_fd:.17g} "
        f"!= JavaScript gradient {bias_grad:.17g}"
    )


def test_relu_zero_uses_zero_derivative() -> None:
    fixture = next(item for item in FIXTURES["cases"] if item["name"] == "relu_zero_convention")
    assert fixture["step"]["zs"][1][0] == 0
    assert fixture["step"]["delta"][1][0] == 0
    assert fixture["step"]["gradW"][0][0][0] == 0
    assert fixture["step"]["gradB"][0][0] == 0

    reference = TorchReference(fixture)
    result = reference.compute(fixture["x"], fixture["y"])
    assert float(result["delta"][1][0]) == 0.0
    assert float(result["gradW"][0][0][0]) == 0.0
    assert float(result["gradB"][0][0]) == 0.0


@pytest.mark.parametrize("trajectory", FIXTURES["trajectories"], ids=lambda item: item["name"])
def test_short_trajectory(trajectory: dict[str, Any]) -> None:
    fixture_shell = {
        "arch": trajectory["arch"],
        "activation": trajectory["activation"],
        "output": trajectory["output"],
        "initialParameters": trajectory["initialParameters"],
    }
    reference = TorchReference(fixture_shell)

    for index, expected in enumerate(trajectory["steps"], start=1):
        prefix = f"{trajectory['name']} step {index}"
        compare_parameters(reference.parameter_snapshot(), expected["beforeParameters"], f"{prefix} before")
        result = reference.compute(expected["x"], expected["y"])
        compare_step(reference, result, expected["step"], trajectory["learningRate"], prefix)
        reference.apply_gradient_descent(trajectory["learningRate"])
        compare_parameters(reference.parameter_snapshot(), expected["afterParameters"], f"{prefix} after")
