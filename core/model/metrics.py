import torch


ZERO = torch.tensor(0)


class MSE():
    def __init__(self, exclude_mask, residual_map=False, batch_mean=False):
        self._mask = ~exclude_mask
        self._residual_map = residual_map
        self._batch_mean = batch_mean

    def __call__(self, pred, true, inp):
        pred = pred[:, self._mask]
        true = true[:, self._mask]
        
        if self._residual_map:
            pred += inp[:, -1, self._mask]

        error = torch.square(pred - true).mean(1)
        if self._batch_mean:
            error = error.mean()
        return error
    
    
class MAE():
    def __init__(self, exclude_mask, residual_map=False, batch_mean=False):
        self._mask = ~exclude_mask
        self._residual_map = residual_map
        self._batch_mean = batch_mean

    def __call__(self, pred, true, inp):
        pred = pred[:, self._mask]
        true = true[:, self._mask]
        
        if self._residual_map:
            pred += inp[:, -1, self._mask]

        error = torch.abs(pred - true).mean(1)
        if self._batch_mean:
            error = error.mean()
        return error


class IOU():
    def __init__(self, contour, exclude_mask, residual_map=False,
                 batch_mean=False):
        self._contour = contour
        self._exclude_mask = exclude_mask
        self._residual_map = residual_map
        self._batch_mean = batch_mean

    def __call__(self, pred, true, inp):
        pred = pred.clone()
        true = true.clone()

        if self._residual_map:
            pred += inp[:, -1]

        pred[:, self._exclude_mask] = 0
        true[:, self._exclude_mask] = 0

        pred_contour = self.get_contour(pred)
        true_contour = self.get_contour(true, True)

        intersection = (pred_contour & true_contour).sum(-1)
        union = (pred_contour | true_contour).sum(-1)

        any_true = true_contour.any(-1)
        iou = torch.where(any_true, 0., torch.nan)
        iou[any_true] = intersection[any_true] / union[any_true]

        if self._batch_mean:
            iou = iou.nanmean(0)
        return iou.squeeze()

    def get_contour(self, data, empty_if_below=False):
        data = data.flatten(1)

        sorted = torch.sort(data, descending=True).values
        contour = self._contour.tile(data.shape[0], 1)

        index = torch.searchsorted(sorted.cumsum(1), contour)
        index[index==data.shape[1]] -= 1

        value = torch.gather(sorted, 1, index)
        if empty_if_below:
            value[value==0] = torch.nan

        return data[:,None] >= value[...,None]
        

class MassConservation():
    def __init__(self, exclude_mask, residual_map=False, batch_mean=False):
        self._mask = ~exclude_mask
        self._residual_map = residual_map
        self._batch_mean = batch_mean
        
    def __call__(self, pred, true, inp):
        pred = pred[:, self._mask]
        true = true[:, self._mask]
        inp = inp[:, -1, self._mask]
        
        if self._residual_map:
            pred_drift_clipped = (pred+inp).maximum(ZERO) - inp
        else:
            pred_drift_clipped = pred.maximum(ZERO) - inp
        true_drift = true - inp
        
        error = pred_drift_clipped.sum(1) - true_drift.sum(1)
        if self._batch_mean:
            error = error.abs().mean()
        return error