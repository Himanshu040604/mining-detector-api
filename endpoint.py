from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
from PIL import Image, ImageDraw
from urllib.parse import unquote_plus
import io, cv2, tempfile, os, numpy as np, time, torch

app = FastAPI(title="Mining Equipment Detector")

# ---------------------------------------------------------------- CONFIG
CLASS_MAP = {
    "Blast rig":    0,
    "Dumper truck": 1,
    "Excavator":    2,
    "car":          3,
}
COLOR_MAP = {
    "Blast rig":    "red",
    "Dumper truck": "blue",
    "Excavator":    "green",
    "car":          "yellow",
}

MODEL_PATH = "final_detection_model.pt"
model = YOLO(MODEL_PATH)
if torch.cuda.is_available():
    model = model.to("cuda")  # use GPU if present

# ---------------------------------------------------------------- HELPERS
def parse_classes(classes: str) -> list[int]:
    valid = {k.lower(): v for k, v in CLASS_MAP.items()}
    ids, bad = [], []
    for raw in classes.split(","):
        name = unquote_plus(raw).strip().lower()
        (ids if name in valid else bad).append(valid.get(name, name))
    if bad:
        raise HTTPException(
            400,
            detail=f"Invalid class(es): {', '.join(bad)}. "
                   f"Choices: {', '.join(CLASS_MAP.keys())}"
        )
    return ids

def detect_and_draw(img: Image.Image, cls_ids: list[int], conf: float) -> Image.Image:
    res = model(img, conf=conf)[0]
    draw = ImageDraw.Draw(img)
    for box in res.boxes:
        cid = int(box.cls)
        if cid in cls_ids:
            name = next(k for k, v in CLASS_MAP.items() if v == cid)
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            draw.rectangle([x1, y1, x2, y2], outline=COLOR_MAP[name], width=3)
    return img

def stream_and_cleanup(out_path: str, in_path: str):
    try:
        with open(out_path, "rb") as f:
            while chunk := f.read(1 << 20):
                yield chunk
    finally:
        for p in (out_path, in_path):
            try:
                os.remove(p)
            except FileNotFoundError:
                pass

# -------------------------------------------------------------- IMAGE ROUTE
@app.post("/detect/image/{classes}", response_class=StreamingResponse)
async def detect_image(
    classes: str,
    file: UploadFile = File(...),
    conf: float = 0.25,
):
    tic = time.time()
    ids = parse_classes(classes)

    if file.filename.split(".")[-1].lower() not in {"jpg", "jpeg", "png"}:
        raise HTTPException(400, "Upload a JPG or PNG image.")

    try:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Could not read image: {e}")

    annotated = detect_and_draw(img, ids, conf)
    buf = io.BytesIO()
    annotated.save(buf, format="JPEG")
    buf.seek(0)

    print(f"✅ Image done in {time.time()-tic:.2f}s [{file.filename}]")
    return StreamingResponse(buf, media_type="image/jpeg")

# -------------------------------------------------------------- VIDEO ROUTE
@app.post("/detect/video/{classes}", response_class=StreamingResponse)
async def detect_video(
    classes: str,
    file: UploadFile = File(...),
    conf: float = 0.25,
):
    tic = time.time()
    ids = parse_classes(classes)

    ext = file.filename.split(".")[-1].lower()
    if ext not in {"mp4", "mov", "avi"}:
        raise HTTPException(400, "Upload an MP4, MOV, or AVI video.")

    tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}")
    tmp_in.write(await file.read())
    tmp_in.close()

    cap = cv2.VideoCapture(tmp_in.name)
    if not cap.isOpened():
        os.remove(tmp_in.name)
        raise HTTPException(400, "Could not open video; unsupported codec?")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ok, _ = cap.read()
    if not ok:
        cap.release()
        os.remove(tmp_in.name)
        raise HTTPException(400, "Failed to decode any frames (re-encode video).")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    writer = cv2.VideoWriter(
        tmp_out.name,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pil = detect_and_draw(pil, ids, conf)
        writer.write(cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR))
        idx += 1
        if idx % 10 == 0:
            print(f"Processed {idx} frames…")

    cap.release()
    writer.release()
    print(f"✅ Video done in {time.time()-tic:.2f}s ({idx} frames)")

    return StreamingResponse(
        stream_and_cleanup(tmp_out.name, tmp_in.name),
        media_type="video/mp4"
    )
