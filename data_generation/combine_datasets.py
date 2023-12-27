import shutil
from pathlib import Path

from tqdm import tqdm
from argparse import ArgumentParser


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=Path, default=Path('/data/fszatkowski/sid/'))
    parser.add_argument('--src_names', type=str, default=['animals', 'sf', 'ameyoko', 'hubble', 'octaves'], nargs="+")
    parser.add_argument('--num_samples_src', type=int, default=10000)
    args = parser.parse_args()

    num_samples_src = args.num_samples_src
    src_names = args.src_names
    data_dir = args.data_dir
    data_paths = list(data_dir.glob(f'*/{num_samples_src}/*/train/dummy'))
    data_paths = [p for p in data_paths if any(name in str(p) for name in src_names)]
    assert len(data_paths) == len(src_names)
    ameyoko = [p for p in data_paths if 'ameyoko' in str(p)][0]
    out_dir = Path(str(ameyoko).replace('ameyoko', 'combined').replace(str(num_samples_src), str(num_samples_src * 5)))

    print()
    print("Copying data to:")
    print(out_dir)
    print()

    img_paths = []
    for data_path in data_paths:
        img_paths.extend(list(data_path.glob('*.jpeg')))
    assert len(img_paths) == num_samples_src * 5, f'Expected {num_samples_src * 5} images, but found only {len(img_paths)}'

    out_dir.mkdir(parents=True, exist_ok=True)
    for i, img_path in tqdm(enumerate(img_paths), f'Copying images to {str(out_dir)}...', total=len(img_paths)):
        out_path = out_dir / f'patch_{i}.jpeg'
        shutil.copyfile(img_path, out_path)
