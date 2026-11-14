import os
import threading
from pathlib import Path
from typing import List, Optional, Tuple

# (x1, y1, x2, y2, confidence)  - pixel coordinates in the original frame
Detection = Tuple[int, int, int, int, float]

_PKG_DIR = Path(__file__).resolve().parent          # .../plate_detection
_ROOT_DIR = _PKG_DIR.parent                         # project folder that holds the package


def _find_model(path: str) -> str:
    """Return a usable model path.

    An absolute path or a path that exists is used as given. Otherwise the file is
    searched for in the current folder, the project root and the package folder, so
    the module still finds best.pt when the team's main app is started from another folder.
    """
    p = Path(path)
    if p.is_absolute() and p.is_file():
        return str(p)
    for base in (Path.cwd(), _ROOT_DIR, _PKG_DIR):
        cand = base / p
        if cand.is_file():
            return str(cand)
    raise FileNotFoundError(
        f"Plate model '{path}' not found. Put best.pt in the project folder or set "
        f"PLATE_MODEL to its full path."
    )


class PlateDetector:
    """YOLO number-plate detector.

    - The model loads lazily on first use, so importing this module is free.
    - Settings can be passed in code or set with environment variables:
        PLATE_MODEL  model file (default best.pt)
        PLATE_CONF   minimum confidence, 0-1 (default 0.25)
        PLATE_IMGSZ  YOLO input size (default 640; try 960 for small/far plates)
    """

    def __init__(self,
                 model_path: Optional[str] = None,
                 conf: Optional[float] = None,
                 imgsz: Optional[int] = None):
        self.model_path = model_path or os.getenv("PLATE_MODEL", "best.pt")
        self.conf = conf if conf is not None else float(os.getenv("PLATE_CONF", "0.25"))
        self.imgsz = imgsz if imgsz is not None else int(os.getenv("PLATE_IMGSZ", "640"))
        self._model = None
        self._lock = threading.Lock()   # FastAPI runs streams in threads; load the model once

    def _load(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from ultralytics import YOLO  # imported here so other modules don't pay for it
                    self._model = YOLO(_find_model(self.model_path))
        return self._model

    def detect(self, frame) -> List[Detection]:
        """Detect plates in one BGR frame (as read by OpenCV). Returns [] if none."""
        if frame is None or getattr(frame, "size", 0) == 0:
            return []
        h, w = frame.shape[:2]
        res = self._load().predict(frame, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
        out: List[Detection] = []
        for b in res.boxes:
            x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
            # keep the box inside the frame and drop empty boxes
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, w), min(y2, h)
            if x2 > x1 and y2 > y1:
                out.append((x1, y1, x2, y2, float(b.conf[0])))
        return out