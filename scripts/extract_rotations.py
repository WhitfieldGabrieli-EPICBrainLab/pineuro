from pathlib import Path
import numpy as np
import pydicom

BASE = Path('examples/example_data/MURFI/feedback')
FILES = [
    BASE / '004_000015_000001_1.3.12.2.1107.5.2.50.176364.2025032717090368237457289.dcm',
    BASE / '004_000015_000002_1.3.12.2.1107.5.2.50.176364.2025032717090462870157653.dcm',
]


def read_motion_tags(dcm_path: Path):
    ds = pydicom.dcmread(dcm_path, force=True)
    if (0x5200, 0x9230) not in ds:
        return [], []
    per_frame = ds[(0x5200, 0x9230)].value
    rot_slices = []
    trans_slices = []
    for frame in per_frame:
        if (0x0021, 0x11FE) in frame:
            motion_seq = frame[(0x0021, 0x11FE)]
            if len(motion_seq.value) > 0:
                motion_item = motion_seq.value[0]
                # Try both tags
                if (0x0021, 0x115A) in motion_item:
                    rot_vals = np.array(motion_item[(0x0021, 0x115A)].value, dtype=float).reshape(-1)
                    rot_slices.append(rot_vals)
                if (0x0021, 0x115B) in motion_item:
                    trans_vals = np.array(motion_item[(0x0021, 0x115B)].value, dtype=float).reshape(-1)
                    trans_slices.append(trans_vals)
    return rot_slices, trans_slices


def summarize(name: str, arrs, convert_radians=False):
    if not arrs:
        print(f"{name}: not found")
        return None
    A = np.vstack(arrs)
    mean_vec = A.mean(axis=0)
    if convert_radians:
        mean_vec = mean_vec * (np.pi / 180.0)
    print(f"{name}: slices={len(arrs)}; first_slice={arrs[0]}; mean={'radians' if convert_radians else 'raw'}={mean_vec}")
    return mean_vec


def main():
    for f in FILES:
        print(f"\nFile: {f}")
        rot_slices, trans_slices = read_motion_tags(f)
        rot_mean = summarize("Tag 0x0021,0x115A (rotation?)", rot_slices, convert_radians=True)
        trans_mean = summarize("Tag 0x0021,0x115B (translation?)", trans_slices, convert_radians=False)
        # Also print a combined motion vector in expected order if both present
        if rot_mean is not None and trans_mean is not None:
            motion_vec = np.concatenate([rot_mean, trans_mean])
            print(f"Combined [rot_x,rot_y,rot_z,trans_x,trans_y,trans_z]: {motion_vec}")


if __name__ == '__main__':
    main()
