#!/usr/bin/env python3


from tqdm import tqdm
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from matplotlib import pyplot as plt
import einops

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import default_collate
from torch.utils.tensorboard import SummaryWriter

from robust_loss_pytorch import AdaptiveLossFunction

from pyproj import Proj, transform

from simple_parsing import Serializable
from file_cache import pickle_file_cache
from with_args import with_args

tqdm.pandas()


def _get_cache_filename(method, self: 'PKDD15Dataset', *args, **kwargs):
    opts = self.opts
    metadata_file = (
        Path(opts.cache_dir).expanduser() /
        '{}-{}.pkl'.format(opts.filename.replace('/', '-'),
                           method.__qualname__)
    )
    return metadata_file


def _parse_polyline(x: str):
    s = x.replace('[', ' ').replace(']', ' ').strip()
    if not s:
        return np.empty(shape=(0, 2), dtype=np.float32)
    else:
        return np.fromstring(s, sep=',').reshape(-1, 2)


def _utm_from_wgs(lon: float, lat: float):
    utm_band = str((np.floor((lon + 180) / 6) % 60) + 1)
    if len(utm_band) == 1:
        utm_band = '0' + utm_band
    if lat >= 0:
        epsg_code = '326' + utm_band
    else:
        epsg_code = '327' + utm_band
    return epsg_code


def _project(x: np.ndarray):
    if x.size <= 0:
        return x
    # x format : lon-lat
    c = x.mean(axis=0)
    lon, lat = c
    # utm_code = _utm_from_wgs(lon, lat)
    p_src = Proj(proj='latlong', datum='WGS84')
    p_dst = Proj(
        "+proj=aeqd +lat_0={} +lon_0={} +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs".format(lat, lon))
    # p_dst = Proj(init='epsg:{}'.format(utm_code))
    out = transform(p_src, p_dst, x[..., 0], x[..., 1])

    return np.stack([out[0], out[1]], axis=-1).astype(np.float32)


class PKDD15Dataset(Dataset):

    @dataclass
    class Settings(Serializable):
        cache_dir: str = '~/.cache/rnn-filter'
        filename: str = '/media/ssd/datasets/taxi-trajectory/train.csv'

    def __init__(self, opts: Settings, transform=None):
        self.opts = opts
        self.transform = transform
        self.data = self._prepare()

    @pickle_file_cache(name_fn=_get_cache_filename)
    def _load(self, filename: str) -> pd.DataFrame:
        df = pd.read_csv(filename)
        return df

    @pickle_file_cache(name_fn=_get_cache_filename)
    def _parse(self, df: pd.DataFrame) -> pd.Series:
        wgs84 = df['POLYLINE'].progress_map(_parse_polyline)
        return wgs84

    @pickle_file_cache(name_fn=_get_cache_filename)
    def _prepare(self) -> pd.Series:
        # csv -> dataframe
        df = self._load(self.opts.filename)

        # extract wgs84 polyline coordinates from dataframe...
        wgs84 = self._parse(df)

        # convert wgs84 coordinates to aeqd w.r.t center.
        aeqd = wgs84.progress_map(_project)

        return aeqd

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # wtf?
        if th.is_tensor(idx):
            idx = idx.tolist()
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample


class SliceInterval(object):
    def __init__(self, sequence_length: int):
        self.seq_len = sequence_length

    def __call__(self, x):
        n = len(x)
        if n < self.seq_len:
            return None
        i0 = np.random.randint(0, n - self.seq_len + 1)
        return x[i0:i0 + self.seq_len]


@dataclass
class Settings(Serializable):
    input_size: int = 2  # x, y
    batch_size: int = 64  # minibatch size
    latent_size: int = 128
    output_size: int = 2  # x, y
    num_layers: int = 2
    sequence_length: int = 8
    min_noise: float = 0.0  # i guess this is like 0m
    max_noise: float = 0.1  # i guess this is like 100m
    num_epochs: int = 16
    dataset: PKDD15Dataset.Settings = PKDD15Dataset.Settings()
    device: str = ''
    log_period: int = 1024
    save_period: int = 8192
    out_dir: str = '/tmp/rnn-filter'
    train: bool = True
    load_file: str = ''
    learning_rate: float = 1e-3


class RnnFilter(nn.Module):
    def __init__(self, opts: Settings):
        super().__init__()
        self.opts = opts

        # input -> latent
        # self.feat = nn.Linear(opts.input_size, opts.latent_size)

        # rnn part
        self.rnn = nn.GRU(
            opts.input_size,
            opts.latent_size,
            opts.num_layers,
            True,
            True)
        # final output
        self.out = nn.Linear(opts.latent_size, opts.output_size, True)

        self.lrelu = nn.LeakyReLU()

    def init_hidden(self, batch_size):
        """
        TODO(ycho): NO idea wtf this is doing.
        """
        weight = next(self.parameters()).data
        hidden = weight.new(
            self.opts.num_layers,
            batch_size,
            self.opts.latent_size).zero_()
        return hidden

    def forward(self, x, h):
        # map to feature
        #x = einops.rearrange(x, 'b t c -> (b t) c')
        #x = self.feat(x)
        #x = einops.rearrange(
        #    x,
        #    '(b t) c -> b t c',
        #    t=self.opts.sequence_length)

        x, h = self.rnn(x, h)

        # NOTE(ycho): Only predicting on very last part
        # y = self.out(self.lrelu(x[:, -1]))
        t = x.shape[1]
        x = einops.rearrange(x, 'b t c -> (b t) c')
        x = self.out(self.lrelu(x))
        x = einops.rearrange(
            x,
            '(b t) c -> b t c',
            t=t)
        return x, h


def _skip_none(batch):
    batch = [x for x in batch if (x is not None)]
    return default_collate(batch)


def _resolve_device(device: str = None):
    # Resolve device ...
    if device:
        device = th.device(device)
    else:
        # o.w. auto resolve --> cuda/cpu
        if th.cuda.is_available():
            device = th.device('cuda:0')
            th.cuda.set_device(device)
        else:
            device = th.device('cpu')
    return device


class RunPath:
    def __init__(self, root: str, key: str = None):
        self.root = Path(root)
        if key is None:
            key = self._resolve_key(root)
        self.dir = self.root / key
        self.dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _resolve_key(root: str) -> str:
        root = Path(root)
        if not root.exists():
            return '0'

        if not root.is_dir():
            raise ValueError('Supplied arg root={} is not a dir'.format(root))

        subdirs = root.glob('*/')
        # NOTE(ycho): Not efficient but convenient
        index = len(list(subdirs))

        while True:
            key = str(index)
            if (root / key).exists():
                index += 1
            break
        return key

    def __getattr__(self, key: str):
        out = self.dir / key
        if not out.is_dir():
            out.mkdir(parents=True, exist_ok=True)
        return out


@with_args()
def main(args: Settings):
    # maybe look into
    # https://www.julienphalip.com/blog/kaggle-competition-report-ecml-pkdd-2015-taxi/
    device = _resolve_device(args.device)

    # == model ==
    model = RnnFilter(args).to(device)
    if args.load_file:
        print('loading ... {}'.format(args.load_file))
        state_dict = th.load(args.load_file)['state']
        model.load_state_dict(state_dict)

    # == dataset ==
    dataset = PKDD15Dataset(
        args.dataset,
        transform=SliceInterval(
            args.sequence_length))
    print('len = {}'.format(len(dataset)))
    loader = DataLoader(dataset, args.batch_size, True, num_workers=8,
                        collate_fn=_skip_none)

    # == train config ==
    # loss_fn = nn.MSELoss()
    loss_obj = AdaptiveLossFunction(
        num_dims=args.output_size,
        float_dtype=th.float32, device=device)
    def loss_fn(output, target):
        loss = th.mean(
            loss_obj.lossfun(
                output.reshape(-1, 2) - target.reshape(-1, 2)
            )
        )
        return loss

    params = list(model.parameters()) + list(loss_obj.parameters())
    optimizer = optim.Adam(params, lr=args.learning_rate)

    # == logging ==
    if args.train:
        path = RunPath(args.out_dir)
        args.save(path.dir / 'args.json')
        writer = SummaryWriter(log_dir=path.log)
        if True:
            # add graph ...
            dummy_x = th.zeros(
                (args.batch_size,
                 args.sequence_length,
                 args.input_size),
                dtype=th.float32,
                device=device)
            dummy_h = th.zeros(
                (args.num_layers,
                 args.batch_size,
                 args.latent_size),
                dtype=th.float32,
                device=device)
            writer.add_graph(model, (dummy_x, dummy_h))

    # == train ==
    try:
        step = 0
        for epoch in range(args.num_epochs):
            net_loss = 0.0
            for i, sample in tqdm(enumerate(loader)):
                step += 1
                # Preprocess sample to be in a decent scale.
                sample = (sample / 1000.0).to(device)

                bs = len(sample)
                h = model.init_hidden(bs).to(device)

                # Create noisy sample ...
                # TODO(ycho): Maybe some samples need to be *not* noisy?
                # Such that the network will adapt to varying noise magnitudes.
                noise_scale = (args.min_noise +
                               (args.max_noise -
                                args.min_noise) *
                               th.rand(bs, dtype=th.float32, device=device))
                z = noise_scale[:, None, None] * th.randn_like(sample)
                x = (sample + z)

                # Prediction + train
                if args.train:
                    out, _ = model(x, h)
                    loss = loss_fn(out, sample)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    net_loss += loss / args.log_period

                    # Log
                    if (i > 0) and (i % args.log_period) == 0:
                        print('net loss = {}'.format(net_loss))
                        writer.add_scalar(
                            'loss', net_loss, global_step=step)
                        net_loss = 0.0

                    # Save
                    if (i > 0) and (i % args.save_period == 0):
                        out_name = 'ckpt-{}-{}.zip'.format(epoch, i)
                        out_file = path.ckpt / out_name
                        th.save({'state': model.state_dict()},
                                str(out_file))
                else:
                    with th.no_grad():
                        out, _ = model(x, h)
                        plt.plot(
                            sample[0, ..., 0].cpu().numpy(),
                            label='sample')
                        plt.plot(
                            x[0, ..., 0].cpu().numpy(),
                            'x', label='noise')
                        plt.plot(out[0, ..., 0].cpu().numpy(), label='output')
                        plt.grid()
                        plt.legend()
                        plt.show()
                        print('{} vs {}'.format(out[0, -1], sample[0, -1]))
    except KeyboardInterrupt:
        pass
    finally:
        writer.close()
        if args.train:
            th.save({'state': model.state_dict()}, path.ckpt / 'model.zip')


if __name__ == '__main__':
    main()
