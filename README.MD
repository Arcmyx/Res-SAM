# GPR Underground Hazard Detection

GPR Underground Hazard Detection adopts a **two-stage processing workflow** to efficiently detect underground hazards and structures. First, the **Segment Anything Model (SAM)** preprocesses the GPR images, rapidly marking potential anomaly candidate regions without requiring additional training, thus improving computational efficiency. Then, **Reservoir Computing (RC)** is used to calculate the anomaly score heatmap. By integrating **Dual-Directional Echo State Networks (2D-ESN) and image patching techniques**, the software refines the anomalies within the candidate regions with high precision.

This software can detect **underground risks** such as **cavities, cracks, and loose regions**, as well as **man-made structures** like **pipes and manholes** (which are not considered risks but still valuable for detection). The anomaly score heatmap highlights high-risk areas for further analysis.

# System Requirements

To ensure smooth operation, the following system specifications are recommended:

- **Operating System**: Linux (Ubuntu 20.04+), Windows 10/11
- **Python Version**: 3.8 or higher
- **GPU Support**: NVIDIA GPU, CUDA 12.4+ recommended for acceleration
- **Memory Requirements**: 16GB RAM or more recommended

# Installation Guide

1. Clone the project repository:

```bash
git clone https://github.com/your-repo/GPR-Underground-Hazard-Detection.git
cd GPR-Underground-Hazard-Detection
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate  # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

If you encounter issues installing PyTorch or Segment Anything Model, please visit their official websites for manual installation:

- [PyTorch](https://pytorch.org/)

- [Segment Anything](https://github.com/facebookresearch/segment-anything)


Additionally, Segment Anything requires downloading the pre-trained model weights: [ViT-L SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth),
which should be placed in the `sam` directory of the project.

# Usage

## Running the Software

Simply run the following command:

```bash
python main.py
```

To switch to the Chinese version, modify the `os.environ['LANGUAGE']` variable in `main.py`:

```python
os.environ['LANGUAGE'] = 'zh-CN'
```

Then, execute the command again:

```bash
python main.py
```

## Software Interface & Operations

![image-20250326095209567](images/image-20250326095209567.png)

1. Select an Image
Click the "Choose an Image" button to load an image from your local storage.

![image-20250326095332711](images/image-20250326095332711.png)

2. Mark Areas of Interest
    - Left-click to mark green points, indicating areas of interest (potential anomaly regions).
    - Right-click to mark red points, indicating non-anomalous areas.
    - The SAM model will use these marked points to draw a blue bounding box around the suspected anomaly.
    - The software will then generate an anomaly score heatmap within this area to visualize risk levels.
    - This step can be repeated multiple times for adjustment.

![image-20250326095522051](images/image-20250326095522051.png)

3. Output Predicted Area

Click "Output Predicted Area" to refine the anomaly regions based on the heatmap. A new red bounding box will be generated.

![image-20250326095916035](images/image-20250326095916035.png)

4. Save Processed Image

Once satisfied with the adjustments, click "Save the Image" to save the processed image locally.

![image-20250326100049407](images/image-20250326100049407.png)

![image-20250326100203941](images/image-20250326100203941.png)

5. Output Image

The saved image includes the processed visualization.

# Data Input & Output

- Input: The software supports PNG and JPEG formats for GPR radar images.

- Output: The processed results can be saved in PNG and JPEG formats.

# Core Modules & API

- `sam/sam.py`: Integrates operations related to the Segment Anything Model (SAM).
    - `predict`: Generates the highest-confidence object mask.
    - `find_box`: Computes the bounding box corresponding to the mask.

- `PatchRes` Module: The core functionality module of the software
    - Computes and visualizes anomaly score heatmaps.
    - Optimizes bounding boxes for detected anomalies.

- `ui` Module: Implements the Graphical User Interface (GUI) to enhance usability.

- `i18n` Module: Provides multi-language translation support.
