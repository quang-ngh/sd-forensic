import lpips
import torch
from torchmetrics.image import PeakSignalNoiseRatio, MultiScaleStructuralSimilarityIndexMeasure

def custom_psnr():
    def compute(preds, target, 
                base=10,
                eps=1e-5,
                reduction="none"):
        """
        args:
            preds: estimated signal/image: [B, C, H, W]
            target: ground truth signal/image: [B, C, H, W]

        """
        assert reduction in ["none", "mean", "sum"]

        preds, target = preds + eps, target + eps
        sqr_error = torch.pow(preds - target, 2)
        numerator = torch.max(target.view(target.shape[0], -1), dim=-1).values
        numerator.unsqueeze_(-1).unsqueeze_(-1).unsqueeze_(-1)
        psnr = sqr_error / (numerator ** 2)

        if isinstance(base, int):
            base = torch.ones_like(numerator, device=numerator.device) * base
        psnr = (10 / torch.log(base)) * (-torch.log(psnr))
        return psnr
    return compute

def register_loss(loss_type, reduction="none"):
    """
    args: loss_type: ["lpips", "l1", "l2"]
    return :
        loss_fn object: 
    """
    loss_fn = None
    assert loss_type in ["lpips", "l1", "l2", "psnr", "ms_ssim"], \
    f"Loss type must be in [lpips, l1, l2]. But found {loss_type}"

    if loss_type == "lpips":
        loss_fn = lpips.LPIPS(net="alex")
    elif loss_type == "l2":
        loss_fn = torch.nn.MSELoss(reduction=reduction)
    elif loss_type == "l1": 
        loss_fn = torch.nn.L1Loss(reduction=reduction)

    elif loss_type == "psnr":
        loss_fn = custom_psnr()

    elif loss_type == "ms_ssim":
        loss_fn = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0, reduction=reduction, 
                                                            return_full_image=True).cuda()
    if loss_fn is None:
        raise ValueError("Cannot create loss object")
    return loss_fn

def print_args(args):
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
# if __name__ == '__main__':
#     psnr_metric = custom_psnr()
#     preds = torch.randn(4,3,16,16).cuda()
#     targets = torch.randn(4,3,16,16).cuda()

#     psnr_score = psnr_metric(preds, targets)