"""16 cols x 8 rows grid of uncurated samples from the official iter-60000 eval set (sample {i}.png has class i % 1000).
128 random indices with distinct classes, seed 0; 2 px white gutters."""
import numpy as np, json
from PIL import Image
SRC = '/localscratch/cva19/AdaIMLE-tmp/eval-official-fd-iter-60000/train/fid'
OUT = '/localscratch/cva19/AdaIMLE/output/rebuttal/fd-iter-60000-grid-16x8'
C, R, S, G = 16, 8, 256, 2
rng = np.random.default_rng(0)
classes = rng.choice(1000, C * R, replace=False)
idx = [int(c + 1000 * rng.integers(50)) for c in classes]          # one random sample of each chosen class
grid = Image.new('RGB', (C * S + (C - 1) * G, R * S + (R - 1) * G), 'white')
for k, i in enumerate(idx):
    grid.paste(Image.open(f'{SRC}/{i}.png').convert('RGB'), ((k % C) * (S + G), (k // C) * (S + G)))
grid.save(OUT + '.png'); grid.save(OUT + '.jpg', quality=95)
grid.resize((grid.width // 4, grid.height // 4), Image.LANCZOS).save(OUT + '-preview.jpg', quality=90)
json.dump(dict(sample_index=idx, class_index=[i % 1000 for i in idx], layout='row-major, 16 cols x 8 rows',
               source=SRC, checkpoint='new-vanilla-results/fd-virtual-incep-lr3e6/iter-60000-model.th'),
          open(OUT + '.json', 'w'), indent=1)
print(grid.size)
