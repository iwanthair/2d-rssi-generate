
import gpytorch


class GaussianProcessModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GaussianProcessModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel() + gpytorch.kernels.PeriodicKernel())
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    

class GaussianProcessModel2(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        k_lp  = gpytorch.kernels.RBFKernel(ard_num_dims=2) * \
                gpytorch.kernels.PeriodicKernel(ard_num_dims=2)
        self.covar_module = gpytorch.kernels.ScaleKernel(k_lp)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GaussianProcessModel3(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        k_lin = gpytorch.kernels.LinearKernel(ard_num_dims=2)
        k_lp  = gpytorch.kernels.RBFKernel(ard_num_dims=2) * \
                gpytorch.kernels.PeriodicKernel(ard_num_dims=2)
        self.covar_module = gpytorch.kernels.ScaleKernel(k_lin * k_lp)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


