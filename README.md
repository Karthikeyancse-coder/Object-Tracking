# AirPods Detection and Object Tracking

This repository contains a complete computer-vision prototype for detecting AirPods in images/video and tracking detections across video frames. It includes a video-frame dataset, a one-class YOLO11 training run, an exploratory multi-class notebook, trained checkpoints, validation/test results, tracking demos, and experiment outputs.

## What was used

| Component | Used for |
|---|---|
| Python 3.11.9 | Notebook experiments and tracking applications (recorded in notebook output) |
| Ultralytics 8.4.173 / YOLO11s | Transfer learning, object detection, evaluation, and tracking API |
| PyTorch 2.11.0+cu128 | Neural-network runtime and CUDA acceleration |
| OpenCV 5.0.0 | Reading videos/camera frames, extracting sampled frames, and drawing/displaying results |
| CUDA GPU | Training and inference; notebook output records an NVIDIA RTX 5070 Laptop GPU and a separate Colab Tesla T4 environment |
| ByteTrack | Tracker used in the AirPods_4 video tracking experiment and default live tracking configuration |
| BoT-SORT | Alternative tracker supported by `tracking.py` |
| YOLO-format labels and YAML | Dataset bounding boxes, class definition, and train/validation/test paths |
| Roboflow export | Source of the 252-image AirPods dataset package; its metadata lists CC BY 4.0 |
| Jupyter / Google Colab | Data preparation, training, evaluation, and experiments |

No `requirements.txt` or other Python dependency lock file is present. The versions above are the ones recorded in notebook output, not a fully reproducible environment specification.

## Dataset and video sources

Four MP4 clips total about 179 seconds and 5,294 frames:

| Video | Resolution | Approx. FPS | Frames | Approx. duration |
|---|---:|---:|---:|---:|
| `AirPods_1.mp4` | 848 × 480 | 30 | 394 | 13 sec |
| `AirPods_2.mp4` | 848 × 480 | 30 | 937 | 32 sec |
| `AirPods_3.mp4` | 848 × 480 | 30 | 2,391 | 81 sec |
| `AirPods_4.mp4` | 480 × 864 | 29 | 1,572 | 53 sec |

The notebook sampled 252 frames. The dataset files currently contain 252 images and 252 non-empty label files, with one class ID (`0`, AirPods) and one bounding-box annotation per image:

| Split | Source video(s) | Images | Label files | Bounding boxes |
|---|---|---:|---:|---:|
| Train | AirPods_1, AirPods_2 | 129 | 129 | 129 |
| Validation | AirPods_3 | 62 | 62 | 62 |
| Test | AirPods_4 | 61 | 61 | 61 |

The split is by source video rather than random frames, reducing near-duplicate leakage across splits. Roboflow metadata says its export resized images to 512 × 512 by stretching and applied no image augmentation at export time. Training then used `imgsz=640`; YOLO also applied the training-time augmentations listed in the saved arguments.

The active local configuration is [`dataset/data.yaml`](dataset/data.yaml), with one class named `AirPods`. Its `path` is an absolute `S:/MODEL/Object/dataset` path and should be changed for another machine. The original Roboflow export is in `export_all/` and the matching archive is `airpods_tracking_v2.v1-airpods_yolo11_v1.yolov11.zip`. The export includes dataset/attribution notes; honor the listed CC BY 4.0 terms when redistributing that data.

## Main AirPods model: YOLO11s version 2

The principal model was initialized from pretrained `yolo11s.pt` and trained for one class. The version 2 settings are recorded in [`airpods_runs/v2/args.yaml`](airpods_runs/v2/args.yaml):

- Detection task, input size 640, batch size 16
- Up to 80 configured epochs; SGD optimizer
- Initial learning rate 0.005, five warmup epochs, patience 40
- GPU device 0, workers 0, AMP disabled
- Standard YOLO augmentations such as mosaic, horizontal flip, scale, and HSV variation

The version 2 [`results.csv`](airpods_runs/v2/results.csv) contains **41 completed/logged epochs**, fewer than the configured maximum of 80. The best observed validation values across those 41 rows are:

| Validation metric | Best logged value | Epoch |
|---|---:|---:|
| Precision | 1.000 | 6 |
| Recall | 0.979 | 2 |
| mAP50 | 0.982 | 4 |
| mAP50–95 | 0.951 | 1 |

The notebook separately evaluates `airpods_runs/v2/weights/best.pt` on the held-out test split and records:

| Test metric | Recorded value |
|---|---:|
| Precision | 0.919 |
| Recall | 0.918 |
| mAP50 | 0.957 |
| mAP50–95 | 0.900 |

These are saved notebook results, not a test rerun performed while preparing this README. The test set contains 61 sampled frames from AirPods_4. Results on one video do not establish broad real-world performance.

Training plots, batch visualizations, confusion matrices, and weights are in [`airpods_runs/v2/`](airpods_runs/v2/). The main inference checkpoint is `airpods_runs/v2/weights/best.pt`; `last.pt` is the final saved checkpoint.

## Recorded full-video tracking experiment

The notebook applies `best.pt` with ByteTrack to all 1,572 frames of `AirPods_4.mp4` at confidence 0.4. Saved notebook output reports a track on all 1,572 frames, 52 distinct track IDs over the clip, and a generated annotated video at `runs/detect/airpods_runs/track_v4/AirPods_4.avi`.

Detected boxes per frame in that recorded run:

| Boxes in frame | Number of frames |
|---:|---:|
| 1 | 966 |
| 2 | 375 |
| 3 | 109 |
| 4 | 65 |
| 5 | 37 |
| 6 | 14 |
| 7 | 5 |
| 8 | 1 |

The 52 IDs indicate tracker identity assignments across the clip, not necessarily 52 physical objects. The notebook reports track ID 1 across all 1,572 frames; other IDs are shorter-lived. This is a useful diagnostic, but the saved outputs do not establish identity accuracy against manually labeled trajectories.

## Exploratory multi-class notebook

[`Object_Tracking.ipynb`](Object_Tracking.ipynb) is a separate Google Colab workflow. It combines AirPods examples with an Open Images subset for ten other classes: Person, Bottle, Mug, Mobile phone, Laptop, Backpack, Headphones, Computer mouse, Book, and Remote control. It trains an 11-class YOLO11s model for 40 configured epochs with image size 640 and batch size 32. The notebook records a class-wise AirPods validation line around precision 1.000, recall 0.919–0.925, mAP50 0.983–0.984, and mAP50–95 0.964–0.965. It also records overall metrics around precision 0.597–0.607, recall 0.443–0.445, mAP50 0.426–0.437, and mAP50–95 0.358–0.361.

This is an exploratory multi-class experiment, not the main single-class model. Its notebook refers to Google Drive/Colab paths, and its training checkpoint is not part of the root project’s named local run folders. `tracking.py` can optionally select class ID 10 for such a multi-class AirPods checkpoint with `--air-class 10`, but the default checkpoint is the one-class version 2 model.

The same notebook describes COCO pretrained classes and YOLO-World free-text detection. The current tracker source supports loading these branches, but the required `yolov8s-worldv2.pt` file is not present in the listed project root; the free-text route therefore needs that model available before use.

## Applications and commands

Run commands from the project root.

### Simple live-camera demo

```powershell
python live_test.py
```

`live_test.py` opens camera index 0 using the Windows DirectShow backend, requests 848 × 480, loads the version 2 `best.pt`, and calls Ultralytics tracking with ByteTrack. It displays FPS, confidence threshold, and the number of detection boxes. Press `q` to quit; `+`/`-` adjusts confidence.

### Configurable tracker

```powershell
python tracking.py airpods
python tracking.py airpods --source clips/AirPods_4.mp4
python tracking.py bottle --log bottle.csv
python tracking.py all --source clips/AirPods_4.mp4 --udp 127.0.0.1:9000
python tracking.py --help
```

`tracking.py` supports the trained AirPods model, COCO classes, multiple requested classes, and YOLO-World text prompts; camera or video sources; ByteTrack/BoT-SORT; selectable target lock and reacquisition; EMA-smoothed pan/tilt/size; a center dead zone; CSV logging; snapshots; and optional UDP JSON publication. UDP is a prototype integration point, not a robot controller. The app chooses CUDA when available and otherwise CPU.

## Folder inventory and purpose

The root inventory at the time of review contained about 1,333 files and 242 MB. Much of the size is video, duplicate image/label exports, training outputs, and model weights.

| Path | Observed contents | Purpose / notes |
|---|---|---|
| `airpods_tracking.ipynb` | 23 cells | Main one-class dataset preparation, YOLO11 training, test evaluation, and video tracking workflow |
| `Object_Tracking.ipynb` | 3,687 lines | Colab/Google Drive multi-class and general object-tracking experiment |
| `clips/` | 4 MP4 files | Original video inputs |
| `frames/` | 252 JPG images | Extracted sample frames organized by video |
| `export_all/` | 252 JPGs, 252 label TXT files, YAML, notes | Original Roboflow YOLO export |
| `dataset/` | 252 JPGs, 252 labels, YAML, 3 cache files | Split train/validation/test dataset used for the single-class model |
| `airpods_runs/v2/` | CSV, args, plots, validation images, `best.pt`, `last.pt` | Main version 2 training artifacts |
| `runs/detect/airpods_runs/v1/` | Earlier training plots, CSV, args, weights | First single-class training experiment |
| `runs/detect/airpods_runs/track_v4/` | Annotated AVI | Output of ByteTrack on AirPods_4 |
| `runs/detect/val/` | Confusion matrices and prediction/label images | Additional validation artifacts |
| `models/best.pt` | Additional checkpoint | Present, but provenance is not documented by a matching local run configuration |
| `weights/yolo26n.pt` | YOLO26n checkpoint | Auxiliary model asset; the recorded custom training used YOLO11s |
| `yolo11s.pt` | YOLO11s checkpoint | Pretrained base weights used for fine-tuning |
| `airpods_tracking_v2...zip` | 509-entry Roboflow package | Compressed dataset export |
| `tracking.py` | Multi-model application | Command-line real-time/video tracker |
| `live_test.py` | Live demo | Minimal camera inference and tracking |
| `Capture.py` | Empty, 0 bytes | Placeholder; contains no implementation |
| `snapshots/` | Empty during inventory | Destination created for tracker snapshots |

The project has no package manifest or automated test suite. `labels.cache` files are YOLO-generated caches, not source annotations.

## What the project demonstrates

- Extracting representative frames from longer video clips and checking dataset coverage
- Working with Roboflow-exported YOLO annotations and a custom dataset YAML
- Splitting video-derived data by source clip
- Transfer-learning a pretrained detector and comparing training configurations
- Separating validation results from a held-out test evaluation
- Interpreting precision, recall, mAP50, and mAP50–95
- Running detection and tracking on recorded video and a live webcam
- Reading track IDs and detection counts to inspect temporal results
- Building target lock/reacquisition, coordinate normalization, smoothing, logs, and messaging into an application
- Comparing a focused one-class detector with an exploratory multi-class detector

## Important caveats and next steps

1. Re-run the test evaluation from the current `best.pt` and save the exact command, package versions, and metrics alongside the run. The metric currently cited is a saved notebook output.
2. The dataset is small and the test split is one video. Add independently recorded clips, lighting conditions, backgrounds, distances, camera angles, and occlusions.
3. Confirm detection and tracking quality visually; track presence in every frame does not prove correct boxes or identity continuity.
4. `dataset/data.yaml` uses a machine-specific absolute path. Change it to a portable relative path or update it for the target environment.
5. Add a dependency lock/requirements file and a clean setup guide. Check GPU availability and camera index on the deployment machine.
6. For robot integration, test latency and watchdog behavior, use safe bounds on control values, and treat invalid target messages as no-action.
7. Respect the Roboflow dataset's CC BY 4.0 attribution terms when sharing the dataset or derived material.
