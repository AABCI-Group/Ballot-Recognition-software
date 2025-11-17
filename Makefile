.PHONY: venv synth split train infer
venv:
	python -m venv .venv && . .venv/bin/activate && \
	pip install --upgrade pip && \
	pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio && \
	pip install -r requirements.txt

synth:
	python src/tools/generate_synthetic.py --blanks_dir data/blanks --stamp_png assets/stamp.png --out_dir data/synth --count 4000

split:
	python src/tools/split_yolo.py --images data/synth/images --labels_det data/synth/labels_det --out_dir data/yolo_split

train:
	python -m src.models.train_yolo --epochs 120 --imgsz 1280 --batch 16

infer:
	python -m src.infer.predict --weights runs/train/yolo_stamp/weights/best.pt --images data/synth/images --out_dir outputs
