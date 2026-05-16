import os
import cv2
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from .io import write_json, safe_mkdir

class DebugWriter:
    """Standardized debug writer for the ballot reader pipeline."""
    
    def __init__(self, out_dir: str, ballot_id: str):
        self.out_dir = os.path.join(out_dir, f"ballot_{ballot_id}")
        safe_mkdir(self.out_dir)

    def get_row_dir(self, row_idx: int) -> str:
        """Get the directory for a specific row."""
        row_dir = os.path.join(self.out_dir, f"row_{row_idx:02d}")
        safe_mkdir(row_dir)
        return row_dir

    def get_box_dir(self, row_idx: int, box_idx: int = 0) -> str:
        """Get the directory for a specific box within a row."""
        box_dir = os.path.join(self.get_row_dir(row_idx), f"box_{box_idx:02d}")
        safe_mkdir(box_dir)
        return box_dir

    def save_image(self, img: np.ndarray, path_parts: List[str], name: str) -> None:
        """Save an image to a standardized path."""
        full_path = os.path.join(self.out_dir, *path_parts, f"{name}.png")
        safe_mkdir(os.path.dirname(full_path))
        cv2.imwrite(full_path, img)

    def save_decision(self, decision: Dict[str, Any], path_parts: List[str]) -> None:
        """Save a decision JSON to a standardized path."""
        full_path = os.path.join(self.out_dir, *path_parts, "decision.json")
        write_json(full_path, decision)

    def save_text(self, text: str, path_parts: List[str], name: str) -> None:
        """Save a text artifact to a standardized path."""
        full_path = os.path.join(self.out_dir, *path_parts, name)
        safe_mkdir(os.path.dirname(full_path))
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(text)
   
   
    def draw_overlay(
        self,
        img: np.ndarray,
        boxes: List[Optional[Tuple[int, int, int, int]]],
        rows: List[Tuple[int, int]],
        name: str,
        assigned: Optional[List[Optional[Tuple[int, int, int, int]]]] = None,
    ) -> None:
        """Draw rows and boxes with a readable overlay. Optionally pass `assigned` to label row↔box."""
        vis = img.copy()
        H, W = vis.shape[:2]

        # --- translucent row fill (so you can see if candidate text is inside)
        overlay = vis.copy()
        for idx, (y1, y2) in enumerate(rows, start=1):
            cv2.rectangle(overlay, (0, y1), (W - 1, y2), (0, 255, 0), thickness=-1)
        vis = cv2.addWeighted(overlay, 0.18, vis, 0.82, 0)

        # --- row outlines + labels
        # Place label near the vote-box column too (right side), not just left
        label_x_left = 10
        label_x_right = int(0.86 * W)
        for idx, (y1, y2) in enumerate(rows, start=1):
            cv2.rectangle(vis, (0, y1), (W - 1, y2), (0, 150, 0), 2)

            cy = (y1 + y2) // 2
            cv2.putText(
                vis, f"R{idx}",
                (label_x_left, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 120, 0), 2, cv2.LINE_AA
            )
            cv2.putText(
                vis, f"R{idx}",
                (label_x_right, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 120, 0), 2, cv2.LINE_AA
            )

        # --- helper: map each box to a row index by center (for labeling/debug)
        def _row_index_for_cy(cy: float) -> Optional[int]:
            for i, (y1, y2) in enumerate(rows):
                if y1 <= cy <= y2:
                    return i
            return None

        # --- draw boxes + center points + labels
        for bidx, box in enumerate(boxes, start=1):
            if box is None:
                if bidx - 1 < len(rows):
                    y1, y2 = rows[bidx - 1]
                    cy = (y1 + y2) // 2
                    cv2.putText(
                        vis, f"B{bidx}=MISSING",
                        (label_x_right, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2, cv2.LINE_AA
                    )
                continue

            x, y, w, h = box
            cy = y + h / 2.0
            cx = x + w / 2.0

            # Determine which row band contains the center
            r_i = _row_index_for_cy(cy)
            in_row = r_i is not None

            # Box color: green if center is inside some row, red if not
            color = (0, 255, 0) if in_row else (0, 0, 255)

            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
            cv2.circle(vis, (int(cx), int(cy)), 3, color, -1)

            # Label above the box
            label = f"B{bidx}"
            if r_i is not None:
                label += f"->R{r_i+1}"
            cv2.putText(
                vis, label,
                (x, max(0, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA
            )

            # If `assigned` was passed, show whether this exact box was used
            if assigned is not None:
                used = any((a is not None and a == (x, y, w, h)) for a in assigned)
                if used:
                    cv2.putText(
                        vis, "USED",
                        (x, min(H - 5, y + h + 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA
                    )

        # Optional: small legend
        cv2.putText(
            vis,
            "Rows: translucent green bands | Boxes: green=OK red=outside-row | dot=center",
            (10, H - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (255, 255, 255), 2, cv2.LINE_AA
        )

        self.save_image(vis, [], name)

    def save_montage(self, images: List[np.ndarray], labels: List[str], name: str) -> None:
        """Save a side-by-side montage of images."""
        if not images:
            return
        
        # Ensure all images are BGR
        bgr_images = []
        for img in images:
            if len(img.shape) == 2:
                bgr_images.append(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
            else:
                bgr_images.append(img)
        
        # Resize all to match the first image's height
        h_target = bgr_images[0].shape[0]
        resized = []
        for img, label in zip(bgr_images, labels):
            h, w = img.shape[:2]
            scale = h_target / h
            new_w = int(w * scale)
            img_res = cv2.resize(img, (new_w, h_target))
            cv2.putText(img_res, label, (10, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            resized.append(img_res)
            
        montage = np.hstack(resized)
        self.save_image(montage, [], name)
