#!/usr/bin/env python3

import unittest

import torch

import Lgpytorch
from Lgpytorch.test.variational_test_case import VariationalTestCase


class TestUnwhitenedVariationalGP(VariationalTestCase, unittest.TestCase):
    @property
    def batch_shape(self):
        return torch.Size([])

    @property
    def distribution_cls(self):
        return Lgpytorch.variational.CholeskyVariationalDistribution

    @property
    def mll_cls(self):
        return Lgpytorch.mlls.VariationalELBO

    @property
    def strategy_cls(self):
        return Lgpytorch.variational.UnwhitenedVariationalStrategy

    def test_training_iteration(self, *args, **kwargs):
        cg_mock, cholesky_mock, ciq_mock = super().test_training_iteration(*args, **kwargs)
        self.assertFalse(cg_mock.called)
        self.assertFalse(ciq_mock.called)
        if self.distribution_cls == Lgpytorch.variational.CholeskyVariationalDistribution:
            self.assertEqual(cholesky_mock.call_count, 3)  # One for each forward pass, once for initialization
        else:
            self.assertEqual(cholesky_mock.call_count, 2)  # One for each forward pass

    def test_eval_iteration(self, *args, **kwargs):
        cg_mock, cholesky_mock, ciq_mock = super().test_eval_iteration(*args, **kwargs)
        self.assertFalse(cg_mock.called)
        self.assertFalse(ciq_mock.called)
        self.assertEqual(cholesky_mock.call_count, 1)  # One to compute cache, that's it!


class TestUnwhitenedPredictiveGP(TestUnwhitenedVariationalGP):
    @property
    def mll_cls(self):
        return Lgpytorch.mlls.PredictiveLogLikelihood


class TestUnwhitenedRobustVGP(TestUnwhitenedVariationalGP):
    @property
    def mll_cls(self):
        return Lgpytorch.mlls.GammaRobustVariationalELBO


class TestUnwhitenedMeanFieldVariationalGP(TestUnwhitenedVariationalGP):
    @property
    def distribution_cls(self):
        return Lgpytorch.variational.MeanFieldVariationalDistribution


class TestUnwhitenedMeanFieldPredictiveGP(TestUnwhitenedPredictiveGP):
    @property
    def distribution_cls(self):
        return Lgpytorch.variational.MeanFieldVariationalDistribution


class TestUnwhitenedMeanFieldRobustVGP(TestUnwhitenedRobustVGP):
    @property
    def distribution_cls(self):
        return Lgpytorch.variational.MeanFieldVariationalDistribution


class TestUnwhitenedDeltaVariationalGP(TestUnwhitenedVariationalGP):
    @property
    def distribution_cls(self):
        return Lgpytorch.variational.DeltaVariationalDistribution


class TestUnwhitenedDeltaPredictiveGP(TestUnwhitenedPredictiveGP):
    @property
    def distribution_cls(self):
        return Lgpytorch.variational.DeltaVariationalDistribution


class TestUnwhitenedDeltaRobustVGP(TestUnwhitenedRobustVGP):
    @property
    def distribution_cls(self):
        return Lgpytorch.variational.DeltaVariationalDistribution


if __name__ == "__main__":
    unittest.main()
