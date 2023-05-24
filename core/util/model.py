class MaskedOutput:
    def __init__(self, exclude_mask):
        self._mask = exclude_mask
        
    def __call__(self, x):
        x[:, self._mask] = 0
        
        return x