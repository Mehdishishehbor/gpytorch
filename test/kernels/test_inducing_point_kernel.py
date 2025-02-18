#!/usr/bin/env python3

import unittest
from unittest.mock import MagicMock, patch

import torch

import Lgpytorch
from Lgpytorch.kernels import InducingPointKernel, RBFKernel, ScaleKernel


class TestModel(Lgpytorch.models.ExactGP):
    def __init__(self, train_x, train_y):
        likelihood = Lgpytorch.likelihoods.GaussianLikelihood()
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = Lgpytorch.means.ZeroMean()
        self.covar_module = InducingPointKernel(
            ScaleKernel(RBFKernel(ard_num_dims=3)),
            inducing_points=torch.randn(512, 3),
            likelihood=likelihood,
        )

    def forward(self, input):
        mean = self.mean_module(input)
        covar = self.covar_module(input)
        return Lgpytorch.distributions.MultivariateNormal(mean, covar)


class TestInducingPointKernel(unittest.TestCase):
    def test_kernel_output(self):
        train_x = torch.randn(1000, 3)
        train_y = torch.randn(1000)
        test_x = torch.randn(500, 3)
        model = TestModel(train_x, train_y)

        # Make sure that the prior kernel is the correct type
        model.train()
        output = model(train_x).lazy_covariance_matrix.evaluate_kernel()
        self.assertIsInstance(output, Lgpytorch.lazy.LowRankRootLazyTensor)

        # Make sure that the prior predictive kernel is the correct type
        model.train()
        output = model.likelihood(model(train_x)).lazy_covariance_matrix.evaluate_kernel()
        self.assertIsInstance(output, Lgpytorch.lazy.LowRankRootAddedDiagLazyTensor)

        # Make sure we're calling the correct prediction strategy
        _wrapped_ps = MagicMock(wraps=Lgpytorch.models.exact_prediction_strategies.SGPRPredictionStrategy)
        with patch("gpytorch.models.exact_prediction_strategies.SGPRPredictionStrategy", new=_wrapped_ps) as ps_mock:
            model.eval()
            output = model.likelihood(model(test_x))
            _ = output.mean + output.variance  # Compute something to break through any lazy evaluations
            self.assertTrue(ps_mock.called)

        # Check whether changing diagonal correction makes a difference (ensuring that cache is cleared)
        model.train()
        model.eval()
        with Lgpytorch.settings.sgpr_diagonal_correction(True), torch.no_grad():
            output_mean_correct = model(test_x).mean
        model.train()
        model.eval()
        with Lgpytorch.settings.sgpr_diagonal_correction(False), torch.no_grad():
            output_mean_no_correct = model(test_x).mean
        self.assertNotAlmostEqual(output_mean_correct.sum().item(), output_mean_no_correct.sum().item())
