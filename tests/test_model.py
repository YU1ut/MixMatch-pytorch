import torch

from mixmatch.models.wideresnet import WideResNet


def test_model_seeded():
    """Test that the model is always initialized the same way when seeded."""

    def create_model(ema=False):
        model_ = WideResNet(num_classes=10)
        model_ = model_.cuda()

        if ema:
            for param in model_.parameters():
                param.detach_()

        return model_

    model_1 = create_model()
    ema_model_1 = create_model(ema=True)
    model_2 = create_model()
    ema_model_2 = create_model(ema=True)

    for param_1, param_2 in zip(model_1.parameters(), model_2.parameters()):
        assert torch.all(torch.eq(param_1, param_2))

    for param_1, param_2 in zip(
        ema_model_1.parameters(), ema_model_2.parameters()
    ):
        assert torch.all(torch.eq(param_1, param_2))
