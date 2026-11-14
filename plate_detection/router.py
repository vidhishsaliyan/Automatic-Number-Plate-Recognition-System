import os, time, uuid
from pathlib import Path
from typing import Callable, List

import cv2                                                    # built-in: OpenCV library
from fastapi import APIRouter, UploadFile, File, HTTPException  # built-in: FastAPI library
from fastapi.responses import StreamingResponse, FileResponse

from .detector import PlateDetector

# All paths/names are namespaced so nothing clashes with other team modules.
DATA_DIR = Path(os.getenv("PLATE_DATA_DIR", "data/plate_detection"))
VIDEO_DIR, CROP_DIR = DATA_DIR / "videos", DATA_DIR / "crops"
FRAME_SKIP = int(os.getenv("PLATE_FRAME_SKIP", "1"))

# self-designed: crop quality limits, chosen after testing on my own videos
MIN_SHARP = float(os.getenv("PLATE_MIN_SHARP", "300"))  # blur threshold (Laplacian variance)
MIN_W = int(os.getenv("PLATE_MIN_W", "50"))             # minimum crop width in pixels
CROP_PAD = 8                                            # margin (pixels) around the plate box
# self-designed: aspect-ratio limits (plates are wide, not square or tall)
MIN_RATIO, MAX_RATIO = 2.0, 6.0

router = APIRouter(prefix="/plate-detection", tags=["plate-detection"])
detector = PlateDetector()

# AI-assisted: hooks let teammates (OCR, vehicle logging) react to each detected plate
# WITHOUT editing this file:  register_hook(fn)
# fn(video_id: str, frame_idx: int, plate_crop: ndarray, bbox: tuple, conf: float)
_hooks: List[Callable] = []


def register_hook(fn: Callable):
    _hooks.append(fn)


def _find_video(vid: str):
    for p in VIDEO_DIR.glob(f"{vid}.*"):
        return p
    return None


def _frames(vid: str, path: Path):
    # AI-assisted: video reading and live streaming loop
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    (CROP_DIR / vid).mkdir(parents=True, exist_ok=True)
    boxes, idx, saved = [], 0, 0
    best, det_frames = 0.0, 0   # for the SUMMARY line at the end
    cands = {}                  # self-designed: best crop for each ~1 second of video
    try:
        while True:
            t0 = time.time()
            ok, frame = cap.read()
            if not ok:
                break
            if idx % FRAME_SKIP == 0:
                # self-designed: keep only wide boxes
                boxes = [b for b in detector.detect(frame)
                         if MIN_RATIO <= (b[2] - b[0]) / max(b[3] - b[1], 1) <= MAX_RATIO]
                if boxes:
                    det_frames += 1
                    best = max(best, max(b[4] for b in boxes))
                fh, fw = frame.shape[:2]
                for x1, y1, x2, y2, c in boxes:
                    # self-designed: padded crop so plate edges are not cut off
                    crop = frame[max(y1 - CROP_PAD, 0):min(y2 + CROP_PAD, fh),
                                 max(x1 - CROP_PAD, 0):min(x2 + CROP_PAD, fw)]
                    if not crop.size or crop.shape[1] < MIN_W:
                        continue  # empty or too small
                    # self-designed: blur check (low Laplacian variance = blurry)
                    sharp = cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
                    if sharp < MIN_SHARP:
                        continue  # not saved, not sent to hooks
                    # self-designed: quality score = confidence x sharpness x width
                    score = c * min(sharp, 3000) * crop.shape[1]
                    # enlarge small sharp crops so OCR can read them
                    if crop.shape[1] < 200:
                        crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                    # self-designed: remember only the best crop of each second
                    win = idx // max(int(fps), 1)
                    if win not in cands or score > cands[win][0]:
                        cands[win] = (score, idx, crop)
                    for hook in _hooks:
                        try:
                            hook(vid, idx, crop, (x1, y1, x2, y2), c)
                        except Exception as e:  # a teammate's bug must not break the stream
                            print("plate hook error:", e)
            # draw boxes and labels on the live video
            for x1, y1, x2, y2, c in boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"plate {c:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                ty = max(y1 - 8, th + 4)
                cv2.rectangle(frame, (x1, ty - th - 4), (x1 + tw + 4, ty + 4), (0, 0, 0), -1)  # dark strip
                cv2.putText(frame, label, (x1 + 2, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
            idx += 1
            time.sleep(max(0, 1 / fps - (time.time() - t0)))
    finally:
        # save the winning crop of each second at high JPEG quality
        for _, (score, i, crop) in cands.items():
            cv2.imwrite(str(CROP_DIR / vid / f"best_f{i:06d}.jpg"), crop,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved += 1
        print(f"SUMMARY video={vid} frames={idx} detected_frames={det_frames} best_conf={best:.2f} crops_saved={saved}")
        cap.release()


# AI-assisted: upload / stream / crops API routes
@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    vid = uuid.uuid4().hex[:8]
    ext = Path(file.filename or "").suffix or ".mp4"
    (VIDEO_DIR / f"{vid}{ext}").write_bytes(await file.read())
    return {"video_id": vid, "stream_url": f"/plate-detection/stream/{vid}"}


@router.get("/stream/{vid}")
def stream(vid: str):
    path = _find_video(vid)
    if not path:
        raise HTTPException(404, "video not found")
    return StreamingResponse(_frames(vid, path),
                             media_type="multipart/x-mixed-replace; boundary=frame")


@router.get("/crops/{vid}")
def list_crops(vid: str):
    d = CROP_DIR / vid
    return sorted(p.name for p in d.glob("*.jpg")) if d.is_dir() else []


@router.get("/crops/{vid}/{name}")
def get_crop(vid: str, name: str):
    p = CROP_DIR / vid / Path(name).name   # .name blocks path traversal
    if not p.is_file():
        raise HTTPException(404)
    return FileResponse(p)