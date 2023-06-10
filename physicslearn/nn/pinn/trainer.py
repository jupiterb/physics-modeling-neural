import numpy as np
import torch as th
from torch.utils.data import Dataset, DataLoader

from physicslearn.equations.abstract import PhyDiffEq
from physicslearn.nn.pinn.abstract import PhysicsInformedNN


class PINNDataset(Dataset):
    def __init__(self, data: th.Tensor, indices: th.Tensor):
        self._data = data
        self._ts_length = data.shape[1] - 1
        self._indices = indices

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, i):
        state_id = self._indices[i]
        ts_id = state_id // self._ts_length
        t = state_id % self._ts_length
        t_tesnor = th.ones((1,)) * t
        X = self._data[ts_id, t]
        Y = self._data[ts_id, t + 1]
        return (X, t_tesnor), Y


class PINNLoss(th.nn.Module):
    """This loss learns model to predict du/dt"""

    def __init__(
        self,
        eq: PhyDiffEq,
        diff_eq_loss_weight: float,
        target_loss_weight: float,
        parameter_loss_w: float,
    ) -> None:
        super(PINNLoss, self).__init__()
        self._eq = eq
        self._eq_w = diff_eq_loss_weight
        self._target_w = target_loss_weight
        self._parameter_w = parameter_loss_w
        self._loss = th.nn.MSELoss()

    def forward(
        self,
        U: th.Tensor,
        U_change: th.Tensor,
        T: th.Tensor,
        parameters: th.Tensor,
        next_U: th.Tensor,
    ):
        batch_size = len(U)
        # compute equation
        dU_dT = th.Tensor(self._eq(U.squeeze(), T, parameters).reshape(U.shape))
        # learn parameters of equation on real data
        mse_parameters = self._loss(dU_dT, next_U - U)
        # useequation to learn U_change
        mse_eq = self._loss(U_change, dU_dT)
        # use real data to learn U_change
        mse_target = self._loss(U_change, next_U - U)
        # compute loss
        return (
            self._eq_w * mse_eq
            + self._target_w * mse_target
            + self._parameter_w * mse_parameters
        ) / batch_size


class PINNTrainer:
    def __init__(
        self,
        data: th.Tensor,
        pinn_loss: PINNLoss,
        train_ratio: float = 0.8,
        batch_size: int = 32,
    ) -> None:
        # TODO create abstract Trainer
        self._pinn_loss = pinn_loss
        self._generate_dataloaders(data, train_ratio, batch_size)

    def fit(
        self,
        pinn: PhysicsInformedNN,
        lr: float,
        epochs: int = 50,
    ) -> PhysicsInformedNN:
        optimizer = th.optim.Adam(pinn.parameters(), lr=lr)
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}\n-------------------------------")
            self._train(pinn, optimizer)
            pinn_loss = self._test(pinn)
            print(f"    PINN test loss: {pinn_loss}")
        return pinn

    def _generate_dataloaders(
        self, data: th.Tensor, train_ratio: float, batch_size: int
    ):
        # split states into train and test
        n_timeseries = data.shape[0]
        ts_length = data.shape[1] - 1
        n_states = n_timeseries * ts_length
        n_train = int(n_states * train_ratio)
        indices = th.randperm(n_states)
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]

        # create datasets
        train_ds = PINNDataset(data, train_indices)
        test_ds = PINNDataset(data, test_indices)

        # create dataloaders
        self._train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        self._test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    def _train(self, pinn: th.nn.Module, pinn_optimizer):
        pinn.train()
        n_batches = len(self._train_dl)
        for i, ((U, T), next_U) in enumerate(self._train_dl):
            # forward by network
            U_change, params = pinn(U, T)

            # compute loss
            pinn_loss = self._pinn_loss(U, U_change, T, params, next_U)

            # backward propagation
            pinn_optimizer.zero_grad()
            pinn_loss.backward()
            pinn_optimizer.step()

            print(f"    {i+1}/{n_batches}: PINN train loss: {pinn_loss}")

    def _test(self, pinn: th.nn.Module):
        pinn.eval()
        pinn_losses = []
        with th.no_grad():
            for (U, T), next_U in self._test_dl:
                # forward by networks
                U_change, params = pinn(U, T)

                # compute losses
                pinn_loss = self._pinn_loss(U, U_change, T, params, next_U)
                pinn_losses.append(pinn_loss)

        mean_pinn_loss = np.mean(pinn_losses)
        return mean_pinn_loss
