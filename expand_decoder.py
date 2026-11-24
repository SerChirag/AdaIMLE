"""Insert extra residual blocks into a trained decoder checkpoint without changing its output.

The new --dec_blocks string may add plain (non-mixin) blocks at the END of any resolution stage
(e.g. "16x3" -> "16x4"). Each added block is a copy of the stage's last trained block with its
last layer (resnet.pw_conv2) zeroed; trained with --plain_residual_blocks set to the printed
indices (residual x + r*branch instead of the convex mix), it is an exact pass-through at step 0.

Usage:
    python expand_decoder.py --old_blocks "<old dec_blocks>" --new_blocks "<new dec_blocks>" \
        --src a-model.th b-model-ema.th --dst_dir <dir>
Writes <dir>/<same file name> for each --src and prints the --plain_residual_blocks value.
"""
import argparse
import os

import torch

from models import parse_layer_string


def stages(blocks):
    """[(res, [indices])] for consecutive runs of the same resolution."""
    out = []
    for i, (res, _) in enumerate(blocks):
        if out and out[-1][0] == res:
            out[-1][1].append(i)
        else:
            out.append((res, [i]))
    return out


def block_map(old_str, new_str):
    """new index -> old index to copy from, and the list of added (pass-through) new indices."""
    old, new = parse_layer_string(old_str), parse_layer_string(new_str)
    old_st, new_st = stages(old), stages(new)
    assert [r for r, _ in old_st] == [r for r, _ in new_st], 'stages must match'
    src, added = {}, []
    for (_, oi), (_, ni) in zip(old_st, new_st):
        assert len(ni) >= len(oi), 'blocks can only be added'
        assert [old[i] for i in oi] == [new[j] for j in ni[:len(oi)]], 'existing blocks must keep their order'
        assert all(new[j][1] is None for j in ni[len(oi):]), 'added blocks must be plain (no mixin)'
        for k, j in enumerate(ni):
            src[j] = oi[min(k, len(oi) - 1)]
            if k >= len(oi):
                added.append(j)
    return src, added


def expand(state_dict, src, added):
    pre = 'decoder.dec_blocks.'
    out = {k: v for k, v in state_dict.items() if not k.startswith(pre)}
    for j, i in src.items():
        for k, v in state_dict.items():
            if k.startswith(f'{pre}{i}.'):
                name = k[len(f'{pre}{i}.'):]
                v = v.clone()
                if j in added and name.startswith('resnet.pw_conv2.'):
                    v.zero_()
                out[f'{pre}{j}.{name}'] = v
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--old_blocks', required=True)
    p.add_argument('--new_blocks', required=True)
    p.add_argument('--src', nargs='+', required=True)
    p.add_argument('--dst_dir', required=True)
    args = p.parse_args()

    src, added = block_map(args.old_blocks, args.new_blocks)
    os.makedirs(args.dst_dir, exist_ok=True)
    for path in args.src:
        sd = torch.load(path, map_location='cpu')
        dst = os.path.join(args.dst_dir, os.path.basename(path))
        torch.save(expand(sd, src, added), dst)
        print(f'{path} -> {dst}')
    print('--plain_residual_blocks', ','.join(map(str, added)))


if __name__ == '__main__':
    main()
