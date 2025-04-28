import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

import environment


class CNP(torch.nn.Module):
    def __init__(self, in_shape, hidden_size, num_hidden_layers, min_std=0.1):
        super(CNP, self).__init__()
        self.d_x = in_shape[0]
        self.d_y = in_shape[1]

        self.encoder = []
        self.encoder.append(torch.nn.Linear(self.d_x + self.d_y, hidden_size))
        self.encoder.append(torch.nn.ReLU())
        for _ in range(num_hidden_layers - 1):
            self.encoder.append(torch.nn.Linear(hidden_size, hidden_size))
            self.encoder.append(torch.nn.ReLU())
        self.encoder.append(torch.nn.Linear(hidden_size, hidden_size))
        self.encoder = torch.nn.Sequential(*self.encoder)

        self.query = []
        self.query.append(torch.nn.Linear(hidden_size + self.d_x, hidden_size))
        self.query.append(torch.nn.ReLU())
        for _ in range(num_hidden_layers - 1):
            self.query.append(torch.nn.Linear(hidden_size, hidden_size))
            self.query.append(torch.nn.ReLU())
        self.query.append(torch.nn.Linear(hidden_size, 2 * self.d_y))
        self.query = torch.nn.Sequential(*self.query)

        self.min_std = min_std

    def nll_loss(self, observation, target, target_truth, observation_mask=None, target_mask=None):
        '''
        The original negative log-likelihood loss for training CNP.
        Parameters
        ----------
        observation : torch.Tensor
            (n_batch, n_context, d_x+d_y) sized tensor that contains context
            points.
            d_x: the number of query dimensions
            d_y: the number of target dimensions.
        target : torch.Tensor
            (n_batch, n_target, d_x) sized tensor that contains query dimensions
            of target (query) points.
            d_x: the number of query dimensions.
            note: n_context and n_target does not need to be the same size.
        target_truth : torch.Tensor
            (n_batch, n_target, d_y) sized tensor that contains target
            dimensions (i.e., prediction dimensions) of target points.
            d_y: the number of target dimensions
        observation_mask : torch.Tensor
            (n_batch, n_context) sized tensor indicating which entries should be
            used in aggregation. Used for batch input.
        target_mask : torch.Tensor
            (n_batch, n_target) sized tensor indicating which entries should be
            used for loss calculation. Used for batch input.
        Returns
        -------
        loss : torch.Tensor (float)
            The NLL loss.
        '''
        mean, std = self.forward(observation, target, observation_mask)
        dist = torch.distributions.Normal(mean, std)
        nll = -dist.log_prob(target_truth)
        if target_mask is not None:
            # sum over the sequence (i.e. targets in the sequence)
            nll_masked = (nll * target_mask.unsqueeze(2)).sum(dim=1)
            # compute the number of entries for each batch entry
            nll_norm = target_mask.sum(dim=1).unsqueeze(1)
            # first normalize, then take an average over the batch and dimensions
            loss = (nll_masked / nll_norm).mean()
        else:
            loss = nll.mean()
        return loss

    def forward(self, observation, target, observation_mask=None):
        '''
        Forward pass of CNP.
        Parameters
        ----------
        observation : torch.Tensor
            (n_batch, n_context, d_x+d_y) sized tensor where d_x is the number
            of the query dimensions, d_y is the number of target dimensions.
        target : torch.Tensor
            (n_batch, n_target, d_x) sized tensor where d_x is the number of
            query dimensions. n_context and n_target does not need to be the
            same size.
        observation_mask : torch.Tensor
            (n_batch, n_context) sized tensor indicating which entries should be
            used in aggregation.
        Returns
        -------
        mean : torch.Tensor
            (n_batch, n_target, d_y) sized tensor containing the mean
            prediction.
        std : torch.Tensor
            (n_batch, n_target, d_y) sized tensor containing the standard
            deviation prediction.
        '''
        h = self.encode(observation)
        r = self.aggregate(h, observation_mask=observation_mask)
        h_cat = self.concatenate(r, target)
        query_out = self.decode(h_cat)
        mean = query_out[..., :self.d_y]
        logstd = query_out[..., self.d_y:]
        std = torch.nn.functional.softplus(logstd) + self.min_std
        return mean, std

    def encode(self, observation):
        h = self.encoder(observation)
        return h

    def decode(self, h):
        o = self.query(h)
        return o

    def aggregate(self, h, observation_mask):
        # this operation is equivalent to taking mean but for
        # batched input with arbitrary lengths at each entry
        # the output should have (batch_size, dim) shape

        if observation_mask is not None:
            h = (h * observation_mask.unsqueeze(2)).sum(dim=1)  # mask unrelated entries and sum
            normalizer = observation_mask.sum(dim=1).unsqueeze(1)  # compute the number of entries for each batch entry
            r = h / normalizer  # normalize
        else:
            # if observation mask is none, we assume that all entries
            # in the batch has the same length
            r = h.mean(dim=1)
        return r

    def concatenate(self, r, target):
        num_target_points = target.shape[1]
        r = r.unsqueeze(1).repeat(1, num_target_points, 1)  # repeating the same r_avg for each target
        h_cat = torch.cat([r, target], dim=-1)
        return h_cat


class Hw5Env(environment.BaseEnv):
    def __init__(self, render_mode="gui") -> None:
        self._render_mode = render_mode
        self.viewer = None
        self._init_position = [0.0, -np.pi/2, np.pi/2, -2.07, 0, 0, 0]
        self._joint_names = [
            "ur5e/shoulder_pan_joint",
            "ur5e/shoulder_lift_joint",
            "ur5e/elbow_joint",
            "ur5e/wrist_1_joint",
            "ur5e/wrist_2_joint",
            "ur5e/wrist_3_joint",
            "ur5e/robotiq_2f85/right_driver_joint"
        ]
        self.reset()
        self._joint_qpos_idxs = [self.model.joint(x).qposadr for x in self._joint_names]
        self._ee_site = "ur5e/robotiq_2f85/gripper_site"

    def _create_scene(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        scene = environment.create_tabletop_scene()
        obj_pos = [0.5, 0.0, 1.5]
        height = np.random.uniform(0.03, 0.1)
        self.obj_height = height
        environment.create_object(scene, "box", pos=obj_pos, quat=[0, 0, 0, 1],
                                  size=[0.03, 0.03, height], rgba=[0.8, 0.2, 0.2, 1],
                                  name="obj1")
        return scene

    def state(self):
        if self._render_mode == "offscreen":
            self.viewer.update_scene(self.data, camera="frontface")
            pixels = torch.tensor(self.viewer.render().copy(), dtype=torch.uint8).permute(2, 0, 1)
        else:
            pixels = self.viewer.read_pixels(camid=0).copy()
            pixels = torch.tensor(pixels, dtype=torch.uint8).permute(2, 0, 1)
            pixels = transforms.functional.center_crop(pixels, min(pixels.shape[1:]))
            pixels = transforms.functional.resize(pixels, (128, 128))
        return pixels / 255.0

    def high_level_state(self):
        ee_pos = self.data.site(self._ee_site).xpos[1:]
        obj_pos = self.data.body("obj1").xpos[1:]
        return np.concatenate([ee_pos, obj_pos, [self.obj_height]])


def bezier(p, steps=100):
    t = np.linspace(0, 1, steps).reshape(-1, 1)
    curve = np.power(1-t, 3)*p[0] + 3*np.power(1-t, 2)*t*p[1] + 3*(1-t)*np.power(t, 2)*p[2] + np.power(t, 3)*p[3]
    return curve


def collect():
    env = Hw5Env(render_mode="gui")
    states_arr = []
    for i in range(100):
        env.reset()
        p_1 = np.array([0.5, 0.3, 1.04])
        p_2 = np.array([0.5, 0.15, np.random.uniform(1.04, 1.4)])
        p_3 = np.array([0.5, -0.15, np.random.uniform(1.04, 1.4)])
        p_4 = np.array([0.5, -0.3, 1.04])
        points = np.stack([p_1, p_2, p_3, p_4], axis=0)
        curve = bezier(points)

        env._set_ee_in_cartesian(curve[0], rotation=[-90, 0, 180], n_splits=100, max_iters=100, threshold=0.05)
        states = []
        for p in curve:
            env._set_ee_pose(p, rotation=[-90, 0, 180], max_iters=10)
            states.append(env.high_level_state())
        states = np.stack(states)
        states_arr.append(states)
        print(f"Collected {i+1} trajectories.", end="\r")

    np.save("bezier_states.npy", states_arr)

    fig, ax = plt.subplots(1, 2)
    for states in states_arr:
        ax[0].plot(states[:, 0], states[:, 1], alpha=0.2, color="b")
        ax[0].set_xlabel("e_y")
        ax[0].set_ylabel("e_z")
        ax[1].plot(states[:, 2], states[:, 3], alpha=0.2, color="r")
        ax[1].set_xlabel("o_y")
        ax[1].set_ylabel("o_z")
    plt.show()
    plt.savefig("bezier_states.png")


def train():
    # load raw trajectories [T x 5]: [ey,ez,oy,oz,height]
    raw_states = np.load("bezier_states.npy", allow_pickle=True)
    # build flat [N x 6]: [t, h, ey, ez, oy, oz]
    S = []
    for traj in raw_states:
        T = traj.shape[0]
        t = np.linspace(0, 1, T).reshape(-1, 1)
        h = traj[:, 4].reshape(-1, 1)
        pos = traj[:, :4]
        S.append(np.concatenate([t, h, pos], axis=1))
    S = np.concatenate(S, axis=0)
    S = torch.tensor(S, dtype=torch.float32)

    # now d_x=2, d_y=4
    model = CNP(in_shape=(2, 4), hidden_size=256, num_hidden_layers=3)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    batch_size = 64
    for i in range(1000):
        idx = torch.randint(0, S.size(0), (batch_size,), dtype=torch.long)
        batch = S[idx]                # [B x 6]
        x_all = batch[:, :2].unsqueeze(0)    # [1, B, 2]
        y_all = batch[:, 2:].unsqueeze(0)    # [1, B, 4]

        # context = all points
        obs = torch.cat([x_all, y_all], dim=-1)  # [1, B, 6]
        tgt_x = x_all                            # [1, B, 2]
        tgt_y = y_all                            # [1, B, 4]

        opt.zero_grad()
        loss = model.nll_loss(obs, tgt_x, tgt_y)
        loss.backward()
        opt.step()

        if i % 100 == 0:
            print(f"Iter {i:4d}, NLL loss = {loss.item():.4f}")

    torch.save(model, "cnp_model.pth")


def evaluate(model, raw_states, n_tests=100, max_context=50, max_target=50):
    """
    raw_states: list of arrays [T x 5] with cols [ey, ez, oy, oz, height]
    """
    errors_end = []
    errors_obj = []

    # build a single [N x 6] array of (t, h, ey, ez, oy, oz)
    states = []
    for traj in raw_states:
        T = traj.shape[0]
        t = np.linspace(0, 1, T).reshape(-1, 1)
        h = traj[:, 4].reshape(-1, 1)
        pos = traj[:, :4]
        states.append(np.concatenate([t, h, pos], axis=1))
    states = np.concatenate(states, axis=0)  # shape [N,6]

    for _ in range(n_tests):
        # random sizes
        n_obs = np.random.randint(1, max_context + 1)
        n_tgt = np.random.randint(1, max_target + 1)
        # sample without replacement
        idx = np.random.choice(len(states), n_obs + n_tgt, replace=False)
        obs = states[idx[:n_obs]]
        tgt = states[idx[n_obs:]]

        # build tensors
        # obs: [n_obs x 6] -> obs_x [1,n_obs,2], obs_y [1,n_obs,4]
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        obs_x = obs[..., :2]
        obs_y = obs[..., 2:]
        observation = torch.cat([obs_x, obs_y], dim=-1)

        # tgt_x: [1,n_tgt,2], tgt_truth [1,n_tgt,4]
        tgt = torch.tensor(tgt, dtype=torch.float32).unsqueeze(0)
        target_x     = tgt[..., :2]
        target_truth = tgt[..., 2:]

        # forward
        mean, _ = model.forward(observation, target_x)
        pred = mean.squeeze(0).detach().numpy()
        true = target_truth.squeeze(0).numpy()

        # split end‐effector vs. object
        mse_end = np.mean((pred[:, :2] - true[:, :2])**2)
        mse_obj = np.mean((pred[:, 2:] - true[:, 2:])**2)

        errors_end.append(mse_end)
        errors_obj.append(mse_obj)

    # plot
    means = [np.mean(errors_obj), np.mean(errors_end)]
    stds  = [np.std(errors_obj),  np.std(errors_end)]
    labels = ['object', 'end-effector']

    plt.figure()
    plt.bar(labels, means, yerr=stds, capsize=5)
    plt.ylabel('MSE')
    plt.title('CNMP Evaluation over 100 random tests')
    plt.savefig('evaluation_errors.png')
    plt.show()

    return means, stds


if __name__ == "__main__":
    # collect()
    #train()

    # load model & data, then evaluate
    raw_states = np.load("bezier_states.npy", allow_pickle=True)
    model = torch.load("cnp_model.pth", weights_only=False)
    evaluate(model, raw_states)


