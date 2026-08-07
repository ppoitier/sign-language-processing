import pytest
import torch

from slp.nn.losses.contrastive.nt_xent import NTXentLoss, nt_xent_loss


def make_views(
    n_views: int = 2,
    batch_size: int = 8,
    dim: int = 16,
    requires_grad: bool = False,
) -> list[torch.Tensor]:
    return [
        torch.randn(batch_size, dim, requires_grad=requires_grad)
        for _ in range(n_views)
    ]


class TestNTXentLoss:

    def test_loss_is_scalar(self):
        views = make_views()
        loss = nt_xent_loss(views)
        assert loss.ndim == 0

    def test_loss_is_finite_and_positive(self):
        views = make_views()
        loss = nt_xent_loss(views)
        assert torch.isfinite(loss)
        assert loss.item() > 0

    def test_differentiable_w_r_t_inputs(self):
        views = make_views(requires_grad=True)
        loss = nt_xent_loss(views)
        loss.backward()
        for view in views:
            assert view.grad is not None
            assert torch.isfinite(view.grad).all()
            assert view.grad.abs().sum() > 0

    def test_module_wrapper_is_differentiable(self):
        views = make_views(requires_grad=True)
        criterion = NTXentLoss(temperature=0.1)
        loss = criterion(views)
        loss.backward()
        assert all(view.grad is not None for view in views)

    def test_perfectly_aligned_views_give_low_loss(self):
        """Identical views (each anchor's positive is a perfect match) should
        be much easier to classify than random noise."""
        base = torch.randn(8, 16)
        aligned_views = [base.clone(), base.clone()]
        random_views = make_views()

        aligned_loss = nt_xent_loss(aligned_views, temperature=0.1)
        random_loss = nt_xent_loss(random_views, temperature=0.1)
        assert aligned_loss.item() < random_loss.item()

    def test_lower_temperature_sharpens_loss(self):
        """With well-aligned positives, a lower temperature should push the
        loss closer to zero (sharper softmax favors the true positive)."""
        base = torch.randn(8, 16)
        views = [base.clone(), base.clone()]
        low_temp_loss = nt_xent_loss(views, temperature=0.05)
        high_temp_loss = nt_xent_loss(views, temperature=1.0)
        assert low_temp_loss.item() < high_temp_loss.item()

    @pytest.mark.parametrize("n_views", [2, 3, 5])
    def test_variable_n_views(self, n_views):
        views = make_views(n_views=n_views, requires_grad=True)
        loss = nt_xent_loss(views)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(view.grad is not None for view in views)

    @pytest.mark.parametrize("batch_size", [1, 2, 16])
    def test_variable_batch_size(self, batch_size):
        views = make_views(batch_size=batch_size)
        loss = nt_xent_loss(views)
        assert torch.isfinite(loss)

    def test_return_accuracy_flag(self):
        views = make_views()
        result = nt_xent_loss(views, return_accuracy=True)
        assert isinstance(result, tuple) and len(result) == 2
        loss, accuracy = result
        assert loss.ndim == 0
        assert accuracy.ndim == 0
        assert 0.0 <= accuracy.item() <= 1.0

    def test_accuracy_does_not_require_grad(self):
        views = make_views(requires_grad=True)
        loss, accuracy = nt_xent_loss(views, return_accuracy=True)
        assert not accuracy.requires_grad
        # The loss must still carry a grad_fn despite accuracy being detached.
        assert loss.requires_grad

    def test_accuracy_is_high_for_well_separated_positives(self):
        """Positives identical and far from all negatives -> near-perfect
        top-1 retrieval."""
        base = torch.randn(8, 16) * 10.0
        views = [base.clone(), base.clone()]
        _, accuracy = nt_xent_loss(views, temperature=0.1, return_accuracy=True)
        assert accuracy.item() == pytest.approx(1.0)

    def test_backward_unaffected_by_return_accuracy(self):
        """Requesting the accuracy alongside the loss should not change the
        loss value or its gradient."""
        torch.manual_seed(0)
        views_a = make_views(requires_grad=True)
        loss_a = nt_xent_loss(views_a)
        loss_a.backward()

        torch.manual_seed(0)
        views_b = make_views(requires_grad=True)
        loss_b, _ = nt_xent_loss(views_b, return_accuracy=True)
        loss_b.backward()

        assert torch.allclose(loss_a, loss_b)
        for a, b in zip(views_a, views_b):
            assert torch.allclose(a.grad, b.grad)
