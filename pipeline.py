import cv2
import shutil
import numpy as np

from pathlib import Path
from ultralytics import YOLO


class YoloPipeline:
    # Инициализация
    def __init__(
        self,
        model_path: str,
        patch_size: int = 640,
        stride: int = 320,
        image_size: tuple = (1152, 31920), # Auto
        work_dir: str = "./work",
    ):
        self.model = YOLO(model_path)
        self.patch_size = patch_size
        self.stride = stride
        self.image_size = image_size

        self.test_patches_dir = Path(work_dir) / "test_patches"                         # Нарезанные изображения
        self.output_pred_dir = Path("C:/Users/User/Desktop/gazprom/runs/segment/work")  # Промежуточные маски
        self.merged_preds_dir = Path(work_dir) / "merged_txt_preds"                     # Результирующие маски

        self.test_patches_dir.mkdir(parents=True, exist_ok=True)
        self.merged_preds_dir.mkdir(parents=True, exist_ok=True)

    def normalize_patch(self, patch):
        if len(patch.shape) == 3 and patch.shape[2] == 3:
            channels = cv2.split(patch)
            eq_channels = [cv2.equalizeHist(c) for c in channels]
            return cv2.merge(eq_channels)
        else:
            return cv2.equalizeHist(patch)

    def split_image(self, image_path: Path):
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        h, w = image.shape[:2]

        for y in range(0, h - self.patch_size + 1, self.stride):
            for x in range(0, w - self.patch_size + 1, self.stride):
                patch = image[y : y + self.patch_size, x : x + self.patch_size]
                patch_name = f"{image_path.stem}_{y}_{x}.png"
                norm_patch = self.normalize_patch(patch)
                cv2.imwrite(str(self.test_patches_dir / patch_name), norm_patch)

    def predict(self):
        self.model.predict(
            source=self.test_patches_dir,
            conf=0.25, # change
            save=False,
            save_txt=True,
            name=self.output_pred_dir,
            exist_ok=True,
        )

    def merge_predictions(self, image_stem: str):
        h, w = self.image_size
        txt_files = list(self.output_pred_dir.rglob(f"{image_stem}_*.txt"))
        merged_lines = []

        for txt_path in txt_files:
            parts = txt_path.stem.split("_")
            y_offset, x_offset = int(parts[-2]), int(parts[-1])
            with open(txt_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    cls = int(parts[0])
                    coords = list(map(float, parts[1:]))
                    xs = coords[::2]
                    ys = coords[1::2]
                    abs_xs = np.array(xs) * self.patch_size + x_offset
                    abs_ys = np.array(ys) * self.patch_size + y_offset
                    norm_xs = abs_xs / w
                    norm_ys = abs_ys / h
                    combined = [cls] + [
                        f"{x:.6f}" for pair in zip(norm_xs, norm_ys) for x in pair
                    ]
                    merged_lines.append(" ".join(map(str, combined)))

        out_txt = self.merged_preds_dir / f"{image_stem}.txt"
        with open(out_txt, "w") as f:
            f.write("\n".join(merged_lines))

    def process_image(self, image_path: Path):
        self.split_image(image_path)
        self.predict()
        self.merge_predictions(image_path.stem)
        self.cleanup() # можно выключить для теста

    def cleanup(self):
        shutil.rmtree(self.test_patches_dir)
        shutil.rmtree(self.output_pred_dir)

    def get_output_dir(self) -> Path:
        return self.merged_preds_dir
