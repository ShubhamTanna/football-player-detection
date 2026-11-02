"""
Football Player Detection: live demo.

Upload a broadcast frame (or pick a sample) and the fine-tuned YOLOv8-nano model finds the players,
goalkeepers, referees and the ball. Players are then split into two teams by shirt colour, and the
team nearest the ball is credited with possession, using the same logic as src/analyse_match.py.

Run locally:  streamlit run app/streamlit_app.py
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.cluster import KMeans
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "runs" / "yolov8n_football" / "weights" / "best.pt"
SAMPLES = sorted((Path(__file__).parent / "samples").glob("*.jpg"))
CLASSES = ["ball", "goalkeeper", "player", "referee"]
# RGB colours for drawing
COL = {"Team A": (242, 245, 240), "Team B": (255, 95, 162), "goalkeeper": (79, 195, 247),
       "referee": (255, 122, 89), "ball": (245, 197, 66), "player": (200, 200, 200)}

st.set_page_config(page_title="Football Player Detection", layout="wide")


@st.cache_resource
def load_model():
    return YOLO(str(WEIGHTS))


def shirt_colour(img_bgr, box):
    x1, y1, x2, y2 = map(int, box)
    w, h = x2 - x1, y2 - y1
    crop = img_bgr[y1 + int(0.18 * h): y1 + int(0.45 * h), x1 + int(0.3 * w): x2 - int(0.3 * w)]
    if crop.size < 6:
        return None
    return np.median(cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3), axis=0)


def analyse(img_rgb, conf, imgsz):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    r = load_model().predict(img_bgr, imgsz=imgsz, conf=conf, device="cpu", verbose=False)[0]
    rows = []
    for box, c, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
        cls = CLASSES[int(c)]
        rows.append(dict(cls=cls, conf=float(s), x1=box[0], y1=box[1], x2=box[2], y2=box[3],
                         colour=shirt_colour(img_bgr, box) if cls == "player" else None))
    dets = pd.DataFrame(rows, columns=["cls", "conf", "x1", "y1", "x2", "y2", "colour"])
    dets["label"] = dets["cls"]

    # Teams: k-means (k=2) on the centre-of-shirt colour of the players in this frame
    pl = dets[dets.cls.eq("player") & dets.colour.notna()]
    if len(pl) >= 4:
        km = KMeans(n_clusters=2, n_init=10, random_state=42).fit(np.stack(pl.colour.values))
        flip = km.cluster_centers_[0, 0] < km.cluster_centers_[1, 0]
        dets.loc[pl.index, "label"] = np.where((km.labels_ == 0) ^ flip, "Team A", "Team B")

    # Possession: the team of the player whose feet are nearest the ball, within 1.5 body-heights
    call = "No ball detected"
    ball = dets[dets.cls.eq("ball")].sort_values("conf")
    teams = dets[dets.label.isin(["Team A", "Team B"])]
    if not ball.empty and not teams.empty:
        b = ball.iloc[-1]
        bx, by = (b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2
        d = np.hypot((teams.x1 + teams.x2) / 2 - bx, teams.y2 - by)
        i = d.idxmin()
        call = teams.loc[i, "label"] if d[i] < 1.5 * (teams.loc[i, "y2"] - teams.loc[i, "y1"]) else "Loose ball"

    out = img_rgb.copy()
    for _, d in dets.iterrows():
        x1, y1, x2, y2 = map(int, (d.x1, d.y1, d.x2, d.y2))
        cx = (x1 + x2) // 2
        col = COL.get(d.label, COL["player"])
        if d.cls == "ball":
            cv2.circle(out, (cx, (y1 + y2) // 2), 14, col, 3)
        else:
            cv2.ellipse(out, (cx, y2), (max((x2 - x1) // 2, 10), 7), 0, -45, 225, col, 3)
    return out, dets, call


st.title("Football player detection")
st.caption("YOLOv8-nano fine-tuned on 298 Bundesliga broadcast frames · test mAP@0.5 0.83 · runs on CPU")

with st.sidebar:
    st.header("Input")
    choice = st.radio("Frame", ["Sample frame", "Upload my own"], label_visibility="collapsed")
    if choice == "Sample frame" and SAMPLES:
        sample = st.selectbox("Sample", SAMPLES, format_func=lambda p: p.stem.replace("_", " ").title())
        img = np.array(Image.open(sample).convert("RGB"))
    else:
        up = st.file_uploader("Broadcast frame (JPG or PNG)", type=["jpg", "jpeg", "png"])
        img = np.array(Image.open(up).convert("RGB")) if up else None
    st.header("Settings")
    conf = st.slider("Confidence threshold", 0.05, 0.9, 0.25, 0.05)
    imgsz = st.select_slider("Inference size (px)", [640, 960, 1280], value=960,
                             help="960 px was chosen on the validation set; it roughly halves missed balls vs 640 px.")

if img is None:
    st.info("Upload a frame from a wide broadcast camera to start.")
    st.stop()

with st.spinner("Detecting…"):
    annotated, dets, call = analyse(img, conf, imgsz)

left, right = st.columns([3, 1])
left.image(annotated, use_container_width=True)
with right:
    st.metric("Possession call", call)
    counts = dets["label"].value_counts()
    for k in ["Team A", "Team B", "goalkeeper", "referee", "ball"]:
        st.write(f"**{k.title() if k.islower() else k}:** {int(counts.get(k, 0))}")
    st.caption("Rings at the feet: white = Team A, pink = Team B, blue = goalkeeper, orange = referee. "
               "Circle = ball.")

with st.expander("How it works and its limits"):
    st.markdown(
        "- **Detection:** YOLOv8-nano (3M parameters) fine-tuned on the Roboflow *football-players-detection* "
        "dataset (CC BY 4.0).\n"
        "- **Teams:** median Lab colour of the centre of each shirt, clustered into 2 groups with k-means. "
        "Team A / Team B labels are arbitrary.\n"
        "- **Possession:** the team of the player whose feet are nearest the ball, if within 1.5 body-heights.\n"
        "- **Limits:** the ball is small (median 8 px wide), so the model finds only about 38% of balls on the "
        "test set. Works best on wide broadcast camera angles like the training data."
    )
