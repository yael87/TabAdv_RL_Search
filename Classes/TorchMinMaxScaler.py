import torch

class TorchMinMaxScaler:
    #The transformation is given by::

    #    X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    #    X_scaled = X_std * (max - min) + min
    
    #min_ : torach.Tensor of shape (n_features,)
    #    Per feature adjustment for minimum. Equivalent to
    #    ``min - X.min(axis=0) * self.scale_``

    #scale_ : torach.Tensor of shape (n_features,)
    #    Per feature relative scaling of the data. Equivalent to
    #    ``(max - min) / (X.max(axis=0) - X.min(axis=0))``
    
    def __init__(
        self,
        feature_range :tuple [int,int] = (0,1),
    ) -> None:
        self.feature_range = feature_range,
        self.min_ = None,
        self.data_min_ = None,
        self.data_max_ = None,
        self.data_range_ = None,
        self.n_samples_seen_ = None, 
        
    def fit(self, x):
        self.data_min_ = torch.min(x, axis=0).values
        self.data_max_ = torch.max(x, axis=0).values

        self.n_samples_seen_ = x.shape[0]

        feature_range = self.feature_range[0]
        
        data_range = torch.subtract(self.data_max_ , self.data_min_)
        self.scale_ = torch.divide((feature_range[1] - feature_range[0]) , data_range)
        self.min_ = torch.multiply(torch.subtract(feature_range[0] , self.data_min_ ), self.scale_)
        self.data_range_ = data_range
        return self
    
    def transform(self, x):
        x = torch.multiply(x, self.scale_)
        x = torch.add(x, self.min_)
        #if self.clip:
        #    np.clip(X, self.feature_range[0], self.feature_range[1], out=X)
        return x
        