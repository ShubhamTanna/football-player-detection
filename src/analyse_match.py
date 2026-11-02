"""
Run the trained football detector on a video or a folder of frames and write an annotated output
with team colours and a per-frame possession call.

Usage
-----
    python src/analyse_match.py --source path/to/clip.mp4 --out outputs/clip_annotated.mp4
    python src/analyse_match.py --source data/test/images --out outputs/test_annotated.mp4

Steps
-----
1. Detect ball / goalkeeper / player / referee with YOLOv8 (inference at 960 px, chosen on validation data).
2. Sample the centre-of-shirt colour of each player and fit k-means (k=2) on the first frames to define two teams.
3. For each frame, the team of the player whose feet are closest to the ball (within 1.5 body-heights)
   gets possession; otherwise the ball is "loose".
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from ultralytics import YOLO

CLASSES = ['ball', 'goalkeeper', 'player', 'referee']
COL = {'A': (240, 245, 242), 'B': (162, 95, 255), 'goalkeeper': (247, 195, 79),
       'referee': (89, 122, 255), 'ball': (66, 197, 245)}          # BGR
ROOT = Path(__file__).resolve().parents[1]


def read_frames(source):
    src = Path(source)
    if src.is_dir():
        for p in sorted(src.glob('*.jpg')) + sorted(src.glob('*.png')):
            yield p.stem, cv2.imread(str(p))
    else:
        cap = cv2.VideoCapture(str(src)); i = 0
        while True:
            ok, frame = cap.read()
            if not ok: break
            yield f'frame_{i:05d}', frame; i += 1
        cap.release()


def shirt_colour(img, box):
    x1, y1, x2, y2 = map(int, box)
    w, h = x2 - x1, y2 - y1
    crop = img[y1 + int(0.18 * h): y1 + int(0.45 * h), x1 + int(0.3 * w): x2 - int(0.3 * w)]
    if crop.size < 6: return None
    return np.median(cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3), axis=0)


def main(a):
    model = YOLO(a.weights)
    frames, dets = [], []
    for name, img in read_frames(a.source):
        r = model.predict(img, imgsz=a.imgsz, conf=a.conf, device=a.device, verbose=False)[0]
        for box, c, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            cls = CLASSES[int(c)]
            dets.append(dict(frame=name, cls=cls, conf=float(s), x1=box[0], y1=box[1], x2=box[2], y2=box[3],
                             colour=shirt_colour(img, box) if cls == 'player' else None))
        frames.append((name, img))
        if a.max_frames and len(frames) >= a.max_frames: break
    dets = pd.DataFrame(dets)
    print(f'{len(frames)} frames, {len(dets)} detections')

    # teams: fit on players from the first frames, then assign everyone
    pl = dets[dets.cls.eq('player') & dets.colour.notna()]
    fit = pl[pl.frame.isin([f for f, _ in frames[:a.fit_frames]])]
    km = KMeans(n_clusters=2, n_init=10, random_state=42).fit(np.stack(fit.colour.values))
    flip = km.cluster_centers_[0, 0] < km.cluster_centers_[1, 0]
    lab = km.predict(np.stack(pl.colour.values))
    dets['team'] = None
    dets.loc[pl.index, 'team'] = np.where((lab == 0) ^ flip, 'A', 'B')

    # possession per frame
    calls = {}
    for name, f in dets.groupby('frame'):
        ball, players = f[f.cls.eq('ball')].sort_values('conf'), f[f.team.notna()]
        if ball.empty or players.empty: calls[name] = 'no ball detected'; continue
        b = ball.iloc[-1]; bx, by = (b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2
        d = np.hypot((players.x1 + players.x2) / 2 - bx, players.y2 - by); i = d.idxmin()
        calls[name] = f'Team {players.loc[i, "team"]}' if d[i] < 1.5 * (players.loc[i, 'y2'] - players.loc[i, 'y1']) else 'loose ball'

    # write video
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0][1].shape[:2]
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*'mp4v'), a.fps, (w, h))
    for name, img in frames:
        for _, d in dets[dets.frame.eq(name)].iterrows():
            x1, y1, x2, y2 = map(int, (d.x1, d.y1, d.x2, d.y2)); cx = (x1 + x2) // 2
            if d.cls == 'ball':
                cv2.circle(img, (cx, (y1 + y2) // 2), 14, COL['ball'], 3)
            else:
                col = COL[d.team] if isinstance(d.team, str) else COL[d.cls if d.cls in COL else 'A']
                cv2.ellipse(img, (cx, y2), (max((x2 - x1) // 2, 10), 7), 0, -45, 225, col, 3)
        cv2.rectangle(img, (0, 0), (560, 64), (29, 42, 15), -1)
        cv2.putText(img, f'Possession: {calls.get(name, "no ball detected")}', (18, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.15, (240, 245, 242), 2, cv2.LINE_AA)
        vw.write(img)
    vw.release()
    share = pd.Series(calls).value_counts(normalize=True).round(3)
    print('Possession share (frames):'); print(share.to_string())
    print(f'Saved {out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source', required=True, help='video file or folder of frames')
    ap.add_argument('--out', default=str(ROOT / 'outputs' / 'annotated.mp4'))
    ap.add_argument('--weights', default=str(ROOT / 'runs' / 'yolov8n_football' / 'weights' / 'best.pt'))
    ap.add_argument('--imgsz', type=int, default=960)
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--fps', type=float, default=25)
    ap.add_argument('--fit-frames', type=int, default=50, help='frames used to learn the two team colours')
    ap.add_argument('--max-frames', type=int, default=0)
    main(ap.parse_args())
