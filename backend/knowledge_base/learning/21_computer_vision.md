# Computer Vision — Interview Preparation Guide

**Candidate:** Rahul Sharma | **Experience:** 4+ years | **Education:** MS Data Science, UMD
**Focus:** Data Scientist / ML Engineer — Computer Vision, Multimodal Systems, Deep Learning
**Hands-on:** Multimodal fashion recommendation (CLIP + DensePose + FAISS), radiology report generation (LLaVA fine-tuning), OpenCV image processing

---

# Table of Contents

1. [Image Fundamentals](#1-image-fundamentals)
2. [Image Preprocessing & Augmentation](#2-image-preprocessing--augmentation)
3. [Classical Computer Vision (OpenCV)](#3-classical-computer-vision-opencv)
4. [CNN Architectures](#4-cnn-architectures)
5. [Object Detection](#5-object-detection)
6. [Semantic & Instance Segmentation](#6-semantic--instance-segmentation)
7. [Pose Estimation](#7-pose-estimation)
8. [Vision Transformers](#8-vision-transformers)
9. [Multimodal Vision-Language Models](#9-multimodal-vision-language-models)
10. [Generative Vision Models](#10-generative-vision-models)
11. [Transfer Learning for Vision](#11-transfer-learning-for-vision)
12. [Your Projects — Deep Dive](#12-your-projects--deep-dive)
13. [Common Interview Questions](#13-common-interview-questions-with-strong-answers)
14. [Key Takeaways](#14-key-takeaways)

---

# **1. Image Fundamentals**

---

## **1.1 Pixels — The Atomic Unit of Images**

A digital image is a 2D grid of pixels. Each pixel stores intensity values that represent color or brightness. Understanding pixel-level operations is the foundation of every CV pipeline.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        IMAGE AS A MATRIX                            │
│                                                                     │
│   Grayscale (H × W × 1):          RGB Color (H × W × 3):          │
│                                                                     │
│   ┌───┬───┬───┬───┐              ┌───┬───┬───┬───┐  ← R channel   │
│   │128│130│135│140│              │255│200│180│160│                  │
│   ├───┼───┼───┼───┤              ├───┼───┼───┼───┤                  │
│   │125│127│133│138│              │240│190│170│150│                  │
│   ├───┼───┼───┼───┤              └───┴───┴───┴───┘                  │
│   │120│122│128│134│              ┌───┬───┬───┬───┐  ← G channel    │
│   └───┴───┴───┴───┘              │100│ 80│ 70│ 60│                  │
│                                   └───┴───┴───┴───┘                  │
│   Values: 0 (black) to            ┌───┬───┬───┬───┐  ← B channel   │
│   255 (white)                     │ 50│ 40│ 30│ 20│                  │
│                                   └───┴───┴───┴───┘                  │
│                                                                     │
│   Shape: (H, W) or (H, W, 1)     Shape: (H, W, 3)                  │
└─────────────────────────────────────────────────────────────────────┘
```

### **Key Concepts**

| Property | Description | Example |
|----------|-------------|---------|
| **Resolution** | H × W pixel count | 1920 × 1080 (Full HD), 224 × 224 (model input) |
| **Bit Depth** | Bits per channel | 8-bit (0–255), 16-bit (0–65535), float32 |
| **Channels** | Color dimensions | 1 (gray), 3 (RGB), 4 (RGBA with alpha) |
| **Aspect Ratio** | Width : Height | 16:9, 4:3, 1:1 (square) |

### **Color Channels**

```
┌──────────────────────────────────────────────────────────────────┐
│                     COLOR CHANNEL BREAKDOWN                       │
│                                                                   │
│  RGB (Red, Green, Blue):                                          │
│    Standard for display and deep learning models                  │
│    pixel = (R, G, B) where each ∈ [0, 255]                      │
│    Example: (255, 0, 0) = pure red                                │
│                                                                   │
│  Grayscale:                                                       │
│    Single intensity channel                                       │
│    gray = 0.299·R + 0.587·G + 0.114·B  (luminance formula)      │
│                                                                   │
│  RGBA:                                                            │
│    RGB + Alpha (transparency) channel                             │
│    Alpha: 0 = fully transparent, 255 = fully opaque              │
│    Used in: PNGs, compositing, segmentation masks                 │
│                                                                   │
│  BGR (OpenCV default):                                            │
│    Blue, Green, Red — reversed channel order                      │
│    cv2.imread() loads as BGR by default                           │
│    Convert: cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                 │
└──────────────────────────────────────────────────────────────────┘
```

---

## **1.2 Resolution, Formats, and Storage**

### **Common Image Formats**

| Format | Compression | Channels | Use Case |
|--------|-------------|----------|----------|
| **JPEG** | Lossy (DCT) | 3 (RGB) | Photos, web images, general CV |
| **PNG** | Lossless (DEFLATE) | 4 (RGBA) | Masks, screenshots, transparency needed |
| **TIFF** | Lossless / None | 1–4+ | Medical imaging (CXR), satellite |
| **DICOM** | Various | 1 (grayscale) | Radiology (your project) |
| **WebP** | Both lossy/lossless | 4 (RGBA) | Web-optimized images |
| **BMP** | None | 3–4 | Raw uncompressed data |

### **Memory Calculation**

For an RGB image of size H × W:

$$
\text{Memory (bytes)} = H \times W \times C \times \text{bytes\_per\_channel}
$$

**Example:** A 1920 × 1080 × 3 (uint8) image uses $1920 \times 1080 \times 3 = 6{,}220{,}800$ bytes ≈ 5.9 MB uncompressed.

### **Data Layout Conventions**

```
┌──────────────────────────────────────────────────────────────────┐
│                     TENSOR DIMENSION ORDERS                       │
│                                                                   │
│   NumPy / OpenCV / PIL:   (H, W, C)    channels-last             │
│     Example: (224, 224, 3)                                        │
│                                                                   │
│   PyTorch / CLIP:         (C, H, W)    channels-first             │
│     Example: (3, 224, 224)                                        │
│                                                                   │
│   Batched PyTorch:        (B, C, H, W)                            │
│     Example: (32, 3, 224, 224)  ← batch of 32 images             │
│                                                                   │
│   TensorFlow (default):   (B, H, W, C)  channels-last            │
│     Example: (32, 224, 224, 3)                                    │
│                                                                   │
│   Conversion:                                                     │
│     img_torch = img_numpy.transpose(2, 0, 1)  # HWC → CHW        │
│     img_torch = torch.from_numpy(img_numpy).permute(2, 0, 1)     │
└──────────────────────────────────────────────────────────────────┘
```

**Interview Tip:** "In my fashion recommendation pipeline, images arrive as PIL (H, W, 3) but CLIP's preprocessor converts them to PyTorch tensors (3, 224, 224) with ImageNet normalization, which we handle inside the CLIP preprocess function."

---

# **2. Image Preprocessing & Augmentation**

---

## **2.1 Resizing Strategies**

Resizing is necessary because models require fixed input dimensions but real-world images vary in resolution and aspect ratio.

```
┌──────────────────────────────────────────────────────────────────────┐
│                      RESIZING APPROACHES                              │
│                                                                       │
│  1. Direct Resize (stretch):                                          │
│     ┌──────────┐      ┌──────┐                                       │
│     │          │  →   │      │   Distorts aspect ratio                │
│     │  640×480 │      │224×  │   Fast but geometry is warped          │
│     │          │      │ 224  │                                        │
│     └──────────┘      └──────┘                                       │
│                                                                       │
│  2. Resize + Center Crop (ImageNet standard):                         │
│     ┌──────────┐  resize   ┌────────┐  crop  ┌──────┐               │
│     │          │  to 256  │        │  224   │      │               │
│     │  640×480 │  ─────→  │ 256×   │ ────→  │224×  │               │
│     │          │  short   │  192   │ center │ 224  │               │
│     └──────────┘  edge    └────────┘        └──────┘               │
│     ✓ Preserves aspect ratio, standard for inference                  │
│                                                                       │
│  3. Resize + Padding (letterbox):                                     │
│     ┌──────────┐      ┌──────┐                                       │
│     │          │  →   │ ░░░░ │  Pads with black/gray                 │
│     │  640×480 │      │ img  │  No info loss, common in YOLO          │
│     │          │      │ ░░░░ │                                        │
│     └──────────┘      └──────┘                                       │
│                                                                       │
│  4. Random Resized Crop (training):                                   │
│     Random scale (0.08–1.0) + random aspect ratio + resize to 224    │
│     ✓ Best augmentation for training CNNs/ViTs                        │
└──────────────────────────────────────────────────────────────────────┘
```

### **Interpolation Methods**

| Method | Speed | Quality | When to Use |
|--------|-------|---------|-------------|
| **Nearest** | Fastest | Blocky artifacts | Masks, segmentation labels |
| **Bilinear** | Fast | Smooth | Default for training |
| **Bicubic** | Medium | Higher quality | Default for inference, ViT |
| **Lanczos** | Slow | Best quality | Final evaluation, publication |
| **Area** | Medium | Good for downscaling | Reducing resolution |

---

## **2.2 Normalization**

### **ImageNet Normalization (The Industry Standard)**

Nearly all pretrained models (ResNet, CLIP, ViT, EfficientNet) were trained with ImageNet statistics:

$$
x_{\text{norm}} = \frac{x - \mu}{\sigma}
$$

| Channel | Mean ($\mu$) | Std ($\sigma$) |
|---------|------|------|
| R | 0.485 | 0.229 |
| G | 0.456 | 0.224 |
| B | 0.406 | 0.225 |

```python
import torchvision.transforms as T

imagenet_normalize = T.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

standard_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),           # Converts PIL [0,255] → float [0,1]
    imagenet_normalize,     # Applies per-channel normalization
])
```

### **Other Normalization Schemes**

| Method | Formula | Use Case |
|--------|---------|----------|
| **[0, 1] scaling** | $x / 255.0$ | Generic float conversion |
| **[-1, 1] scaling** | $(x / 255.0) \times 2 - 1$ | GANs, Stable Diffusion |
| **ImageNet** | $(x - \mu) / \sigma$ | All ImageNet-pretrained models |
| **CLIP-specific** | Built into `clip.load()` preprocess | CLIP models (your project) |
| **Per-image** | $(x - \text{mean}(x)) / \text{std}(x)$ | Medical imaging, custom domains |

**Interview Tip:** "In our fashion recommendation system, CLIP's preprocess function handles normalization internally — it resizes to 224×224, converts to tensor, and applies CLIP-specific normalization. We don't manually normalize."

---

## **2.3 Data Augmentation**

Augmentation artificially increases training data diversity to reduce overfitting and improve generalization.

```
┌────────────────────────────────────────────────────────────────────────┐
│                    DATA AUGMENTATION TAXONOMY                           │
│                                                                        │
│   GEOMETRIC (spatial):                                                 │
│   ├── Random Horizontal Flip (p=0.5) ← most common                   │
│   ├── Random Vertical Flip (p=0.5)   ← medical, satellite             │
│   ├── Random Rotation (±15° to ±30°)                                  │
│   ├── Random Resized Crop (scale 0.08–1.0)                            │
│   ├── Random Affine (translate, shear, scale)                         │
│   └── Random Perspective                                               │
│                                                                        │
│   PHOTOMETRIC (color/intensity):                                       │
│   ├── Color Jitter (brightness, contrast, saturation, hue)            │
│   ├── Gaussian Blur (kernel 3–7)                                      │
│   ├── Random Grayscale (p=0.1–0.2)                                    │
│   ├── Random Solarization                                              │
│   └── Random Posterization                                             │
│                                                                        │
│   ERASING / REGULARIZATION:                                            │
│   ├── Cutout — erase a random rectangular patch                       │
│   ├── Random Erasing — erase with random fill values                  │
│   ├── GridMask — erase grid-patterned regions                         │
│   └── CutMix — replace patch with patch from another image            │
│                                                                        │
│   MIXING (sample-level):                                               │
│   ├── Mixup — blend two images + labels:                              │
│   │     x̃ = λ·x₁ + (1-λ)·x₂,  ỹ = λ·y₁ + (1-λ)·y₂                │
│   ├── CutMix — paste patch from one image onto another                │
│   └── Mosaic (YOLO) — stitch 4 images into one                       │
│                                                                        │
│   AUTO-AUGMENTATION:                                                   │
│   ├── AutoAugment — learned policies from ImageNet                    │
│   ├── RandAugment — N random transforms, magnitude M                  │
│   └── TrivialAugment — single random transform per image              │
└────────────────────────────────────────────────────────────────────────┘
```

### **Standard Training Pipeline (ImageNet-style)**

```python
train_transform = T.Compose([
    T.RandomResizedCrop(224, scale=(0.08, 1.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    T.RandomGrayscale(p=0.2),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    T.RandomErasing(p=0.25, scale=(0.02, 0.33)),  # Cutout-style
])

val_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

### **Mixup and CutMix — Mathematical Formulation**

**Mixup** (Zhang et al., 2018):

$$
\tilde{x} = \lambda x_i + (1 - \lambda) x_j, \quad \tilde{y} = \lambda y_i + (1 - \lambda) y_j
$$

where $\lambda \sim \text{Beta}(\alpha, \alpha)$, typically $\alpha = 0.2$.

**CutMix** (Yun et al., 2019):

$$
\tilde{x} = \mathbf{M} \odot x_i + (1 - \mathbf{M}) \odot x_j, \quad \tilde{y} = \lambda y_i + (1 - \lambda) y_j
$$

where $\mathbf{M}$ is a binary mask indicating the cut region, and $\lambda = 1 - \frac{r_w \cdot r_h}{W \cdot H}$ (proportion of area cut).

### **Augmentation Selection Guide**

| Task | Recommended Augmentations |
|------|---------------------------|
| **Classification** | RandomResizedCrop, Flip, ColorJitter, RandAugment, Mixup/CutMix |
| **Object Detection** | Mosaic, RandomAffine, HSV augment, Flip (YOLO-style) |
| **Segmentation** | Joint spatial transforms (same transform for image + mask) |
| **Medical Imaging** | Rotation, Flip (both axes), Elastic deformation, Intensity shift |
| **Contrastive Learning** | Strong augmentation: crop + color + blur + grayscale + solarize |

---

# **3. Classical Computer Vision (OpenCV)**

---

## **3.1 Color Spaces**

```
┌─────────────────────────────────────────────────────────────────────┐
│                      COLOR SPACE CONVERSIONS                         │
│                                                                      │
│   RGB ←→ BGR         cv2.cvtColor(img, cv2.COLOR_RGB2BGR)           │
│                      OpenCV default is BGR, PIL default is RGB       │
│                                                                      │
│   RGB → Grayscale    gray = 0.299R + 0.587G + 0.114B               │
│                      cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)           │
│                                                                      │
│   RGB → HSV          Hue (0–180), Saturation (0–255), Value (0–255) │
│                      ✓ Isolate color (H) from brightness (V)         │
│                      ✓ Robust to lighting changes                    │
│                      Use: skin detection, object tracking by color   │
│                                                                      │
│   RGB → LAB          L* (lightness), a* (green–red), b* (blue–yel) │
│                      ✓ Perceptually uniform: ΔE ≈ perceived diff    │
│                      Use: color correction, histogram equalization    │
│                                                                      │
│   RGB → YCrCb        Y (luma), Cr (red diff), Cb (blue diff)       │
│                      Use: face detection, compression (JPEG uses YCbCr) │
└─────────────────────────────────────────────────────────────────────┘
```

```python
import cv2
import numpy as np

img_bgr = cv2.imread("fashion_item.jpg")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# HSV-based color filtering (isolate red clothing)
lower_red = np.array([0, 120, 70])
upper_red = np.array([10, 255, 255])
mask = cv2.inRange(img_hsv, lower_red, upper_red)
result = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
```

---

## **3.2 Thresholding**

Thresholding converts a grayscale image into a binary image — the simplest form of segmentation.

| Method | Description | When to Use |
|--------|-------------|-------------|
| **Simple (Binary)** | `pixel > T ? 255 : 0` | Uniform lighting, controlled environment |
| **Otsu's** | Automatically finds optimal T by minimizing intra-class variance | Bimodal histogram (clear foreground/background) |
| **Adaptive Mean** | T = local mean in neighborhood | Varying illumination across image |
| **Adaptive Gaussian** | T = Gaussian-weighted local mean | Better than mean for noise |
| **Triangle** | Finds T from histogram geometry | Unimodal distribution |

```python
# Otsu's automatic thresholding
_, binary_otsu = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Adaptive Gaussian thresholding
binary_adaptive = cv2.adaptiveThreshold(
    img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, blockSize=11, C=2
)
```

---

## **3.3 Edge Detection**

### **Canny Edge Detector — The Gold Standard**

Canny is a multi-stage algorithm producing clean, thin edges:

```
┌────────────────────────────────────────────────────────────────┐
│                   CANNY EDGE DETECTION PIPELINE                 │
│                                                                 │
│   Step 1: Gaussian Blur                                         │
│     Smooth the image to reduce noise                            │
│     G(x,y) = (1/2πσ²) · exp(-(x²+y²)/2σ²)                    │
│                                                                 │
│   Step 2: Gradient Computation (Sobel)                          │
│     Compute magnitude and direction:                            │
│     |G| = √(Gx² + Gy²),  θ = arctan(Gy/Gx)                   │
│                                                                 │
│   Step 3: Non-Maximum Suppression (NMS)                         │
│     Keep only pixels that are local maxima along gradient dir   │
│     → Produces thin (1-pixel-wide) edges                        │
│                                                                 │
│   Step 4: Double Thresholding                                   │
│     Strong edges: gradient > T_high                             │
│     Weak edges: T_low < gradient < T_high                       │
│     Non-edges: gradient < T_low                                 │
│                                                                 │
│   Step 5: Edge Tracking by Hysteresis                           │
│     Keep weak edges only if connected to strong edges           │
│     → Eliminates isolated noise while preserving true edges     │
└────────────────────────────────────────────────────────────────┘
```

### **Sobel Operator**

Computes image gradients using convolution kernels:

$$
G_x = \begin{bmatrix} -1 & 0 & +1 \\ -2 & 0 & +2 \\ -1 & 0 & +1 \end{bmatrix} * I, \quad
G_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ +1 & +2 & +1 \end{bmatrix} * I
$$

```python
# Canny edge detection
edges = cv2.Canny(img_gray, threshold1=50, threshold2=150)

# Sobel gradients
sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)

# Laplacian (second derivative — detects edges as zero crossings)
laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
```

---

## **3.4 Contour Detection**

Contours are curves joining continuous points with the same intensity — they represent object boundaries.

```python
# Find contours
contours, hierarchy = cv2.findContours(
    binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

# Filter by area (remove noise)
min_area = 500
filtered = [c for c in contours if cv2.contourArea(c) > min_area]

# Bounding box extraction
for contour in filtered:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(img_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Contour properties
area = cv2.contourArea(contour)
perimeter = cv2.arcLength(contour, closed=True)
circularity = (4 * np.pi * area) / (perimeter ** 2)
```

### **Retrieval Modes**

| Mode | Description |
|------|-------------|
| `RETR_EXTERNAL` | Only outermost contours (no children) |
| `RETR_LIST` | All contours, no hierarchy |
| `RETR_TREE` | Full hierarchy tree |
| `RETR_CCOMP` | Two-level hierarchy (outer + inner holes) |

---

## **3.5 Morphological Operations**

Morphological ops process binary images using a structuring element (kernel):

```
┌──────────────────────────────────────────────────────────────────┐
│                  MORPHOLOGICAL OPERATIONS                          │
│                                                                   │
│   Erosion:     Shrinks white regions (removes small noise)        │
│     dst(x,y) = min over kernel of src(x+kx, y+ky)               │
│                                                                   │
│   Dilation:    Expands white regions (fills small holes)          │
│     dst(x,y) = max over kernel of src(x+kx, y+ky)               │
│                                                                   │
│   Opening:     Erosion → Dilation (remove noise, keep shape)      │
│     ✓ Removes small bright spots on dark background               │
│                                                                   │
│   Closing:     Dilation → Erosion (fill holes, keep shape)        │
│     ✓ Fills small dark holes in bright objects                    │
│                                                                   │
│   Gradient:    Dilation − Erosion (extract boundary)              │
│     ✓ Produces outline of objects                                  │
│                                                                   │
│   Top Hat:     Original − Opening (bright detail extraction)      │
│   Black Hat:   Closing − Original (dark detail extraction)        │
└──────────────────────────────────────────────────────────────────┘
```

```python
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

eroded  = cv2.erode(binary_img, kernel, iterations=1)
dilated = cv2.dilate(binary_img, kernel, iterations=1)
opened  = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
closed  = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
```

---

## **3.6 Template Matching**

Slides a template image across the source image and computes similarity at each position.

```python
result = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# TM_CCOEFF_NORMED: best match at maximum value
top_left = max_loc
h, w = template.shape[:2]
cv2.rectangle(img_bgr, top_left, (top_left[0]+w, top_left[1]+h), (0,255,0), 2)
```

**Limitations:** Not scale-invariant or rotation-invariant. For robust matching, use feature-based methods.

---

## **3.7 Feature Detection (SIFT, SURF, ORB)**

### **Comparison Table**

| Feature | SIFT | SURF | ORB |
|---------|------|------|-----|
| **Full Name** | Scale-Invariant Feature Transform | Speeded-Up Robust Features | Oriented FAST + Rotated BRIEF |
| **Keypoint** | DoG (Difference of Gaussians) | Hessian matrix approximation | FAST corners |
| **Descriptor** | 128-D float | 64-D float | 256-bit binary |
| **Scale Invariant** | Yes | Yes | Limited |
| **Rotation Invariant** | Yes | Yes | Yes |
| **Speed** | Slow | Medium | Very fast |
| **Patent** | Expired (2020) | Patented | Free / open |
| **Matching** | L2 distance | L2 distance | Hamming distance |
| **Best For** | Accuracy-critical | Balance speed/accuracy | Real-time, mobile |

```python
# ORB (free and fast)
orb = cv2.ORB_create(nfeatures=500)
kp, des = orb.detectAndCompute(img_gray, None)

# SIFT (patent expired 2020 — now available in OpenCV)
sift = cv2.SIFT_create()
kp, des = sift.detectAndCompute(img_gray, None)

# Feature matching with BFMatcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  # For ORB
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)
```

---

# **4. CNN Architectures**

---

## **4.1 Evolution Overview**

```
┌────────────────────────────────────────────────────────────────────────┐
│                      CNN ARCHITECTURE EVOLUTION                         │
│                                                                        │
│   1998    LeNet-5       Yann LeCun — digit recognition (60K params)   │
│     │                   Conv → Pool → Conv → Pool → FC                │
│     ▼                                                                  │
│   2012    AlexNet       ImageNet winner (60M params)                   │
│     │                   ReLU, Dropout, GPU training, data augmentation │
│     ▼                                                                  │
│   2014    VGGNet        Deep uniform 3×3 convs (138M params)          │
│     │                   "Deeper is better" — VGG-16, VGG-19           │
│     ▼                                                                  │
│   2014    GoogLeNet     Inception module — multi-scale features        │
│     │                   1×1, 3×3, 5×5 parallel branches (6.8M params) │
│     ▼                                                                  │
│   2015    ResNet        Skip connections solve vanishing gradients     │
│     │                   152 layers, won ImageNet (25.6M for ResNet-50)│
│     ▼                                                                  │
│   2017    DenseNet      Dense connections — every layer connects to    │
│     │                   every subsequent layer (feature reuse)         │
│     ▼                                                                  │
│   2019    EfficientNet  Compound scaling (depth × width × resolution) │
│     │                   NAS-designed, SOTA efficiency (5.3M for B0)    │
│     ▼                                                                  │
│   2017+   MobileNet     Depthwise separable convolutions              │
│                         Designed for mobile/edge (3.4M for V2)        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## **4.2 Key Innovations Per Architecture**

### **ResNet — The Most Important Innovation (Skip Connections)**

The vanishing gradient problem made training networks beyond ~20 layers impractical. ResNet solves this with **residual connections**:

$$
\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x}
$$

```
┌────────────────────────────────────────────────────────┐
│                   RESIDUAL BLOCK                        │
│                                                         │
│    x ─────────────────────────────┐                     │
│    │                              │ (identity shortcut)  │
│    ▼                              │                     │
│  [Conv 3×3] → [BN] → [ReLU]      │                     │
│    │                              │                     │
│    ▼                              │                     │
│  [Conv 3×3] → [BN]               │                     │
│    │                              │                     │
│    ▼                              ▼                     │
│  [ + ] ←──────────────────────────┘                     │
│    │                                                    │
│    ▼                                                    │
│  [ReLU]                                                 │
│    │                                                    │
│    ▼                                                    │
│    y = F(x) + x                                         │
│                                                         │
│  Key Insight: The network only needs to learn the       │
│  RESIDUAL F(x) = H(x) - x, not the full mapping H(x). │
│  If identity is optimal, F(x) → 0 is easy to learn.    │
└────────────────────────────────────────────────────────┘
```

### **Bottleneck Block (ResNet-50+)**

Uses 1×1 → 3×3 → 1×1 convolutions to reduce computation:

```
Input (256 channels)
  │
  ├──→ [1×1 Conv, 64]  → [BN] → [ReLU]    ← Reduce dimensions
  │
  ├──→ [3×3 Conv, 64]  → [BN] → [ReLU]    ← Process features
  │
  ├──→ [1×1 Conv, 256] → [BN]              ← Restore dimensions
  │
  └──────────────────────────+ (add)
                             │
                           [ReLU]
```

### **Inception Module (GoogLeNet)**

Processes features at **multiple scales simultaneously**:

```
┌───────────────────────────────────────────────────────────────┐
│                    INCEPTION MODULE                             │
│                                                                │
│                      Input                                      │
│            ┌────┬────┬────┬────┐                               │
│            │    │    │    │    │                                │
│            ▼    ▼    ▼    ▼    │                                │
│          [1×1] [1×1] [1×1] [3×3 MaxPool]                       │
│            │    │    │    │                                     │
│            │    ▼    ▼    ▼                                     │
│            │  [3×3] [5×5] [1×1]                                │
│            │    │    │    │                                     │
│            └────┴────┴────┘                                     │
│                  │                                              │
│           [Concatenate along channel dim]                       │
│                                                                │
│   1×1 convs act as "bottlenecks" to reduce channel count       │
│   before expensive 3×3 and 5×5 operations.                     │
└───────────────────────────────────────────────────────────────┘
```

### **EfficientNet — Compound Scaling**

Instead of arbitrarily scaling depth, width, or resolution independently, EfficientNet scales all three together:

$$
\text{depth:} \quad d = \alpha^\phi, \quad \text{width:} \quad w = \beta^\phi, \quad \text{resolution:} \quad r = \gamma^\phi
$$

subject to $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$, where $\phi$ is the compound coefficient.

| Variant | Params | Top-1 Accuracy | Resolution |
|---------|--------|----------------|------------|
| **B0** | 5.3M | 77.3% | 224 |
| **B3** | 12M | 81.6% | 300 |
| **B7** | 66M | 84.3% | 600 |

### **MobileNet — Depthwise Separable Convolutions**

Standard conv: $D_K \times D_K \times M \times N$ multiplications.
Depthwise separable: $D_K \times D_K \times M + M \times N$ — up to **8–9× reduction**.

```
┌────────────────────────────────────────────────────────────────────┐
│              DEPTHWISE SEPARABLE CONVOLUTION                        │
│                                                                     │
│   Standard Conv (3×3, 32→64):    3×3×32×64 = 18,432 params        │
│                                                                     │
│   Depthwise Separable:                                              │
│     Step 1 — Depthwise: 3×3 conv per channel   3×3×32   = 288     │
│     Step 2 — Pointwise: 1×1 conv across chans  1×1×32×64 = 2,048  │
│     Total: 2,336 params  (7.9× reduction)                          │
│                                                                     │
│   Computation savings:                                              │
│     1/N + 1/D_K² ≈ 1/64 + 1/9 ≈ 0.127  (87% fewer FLOPs)        │
└────────────────────────────────────────────────────────────────────┘
```

### **Quick Reference Table**

| Architecture | Year | Params | Top-1 (ImageNet) | Key Innovation |
|-------------|------|--------|-------------------|----------------|
| LeNet-5 | 1998 | 60K | — | First practical CNN |
| AlexNet | 2012 | 60M | 63.3% | ReLU, Dropout, GPU training |
| VGG-16 | 2014 | 138M | 74.5% | Uniform 3×3 convs, depth |
| GoogLeNet | 2014 | 6.8M | 74.8% | Inception (multi-scale) |
| ResNet-50 | 2015 | 25.6M | 76.1% | Skip connections |
| DenseNet-121 | 2017 | 8M | 75.0% | Dense connectivity |
| EfficientNet-B0 | 2019 | 5.3M | 77.3% | Compound scaling (NAS) |
| MobileNetV2 | 2018 | 3.4M | 72.0% | Inverted residuals + linear bottleneck |

---

# **5. Object Detection**

---

## **5.1 Problem Definition**

Object detection = **classification + localization**. For each object in an image, predict:
- **Class label** (what is it?)
- **Bounding box** (where is it?) — $(x_{\min}, y_{\min}, x_{\max}, y_{\max})$ or $(x_c, y_c, w, h)$

```
┌──────────────────────────────────────────────────────────────────┐
│                  OBJECT DETECTION TAXONOMY                         │
│                                                                   │
│   TWO-STAGE DETECTORS (accuracy-focused):                         │
│   ├── R-CNN (2014) — selective search + CNN                       │
│   ├── Fast R-CNN (2015) — shared CNN features + RoI pooling       │
│   ├── Faster R-CNN (2015) — Region Proposal Network (RPN)         │
│   └── Mask R-CNN (2017) — Faster R-CNN + instance segmentation    │
│                                                                   │
│   ONE-STAGE DETECTORS (speed-focused):                            │
│   ├── SSD (2016) — multi-scale feature maps                      │
│   ├── YOLO v1 (2016) — single-shot grid prediction               │
│   ├── YOLO v3 (2018) — multi-scale, FPN, better accuracy         │
│   ├── YOLO v5 (2020) — PyTorch, ultralytics, practical           │
│   └── YOLO v8 (2023) — anchor-free, decoupled head               │
│                                                                   │
│   ANCHOR-FREE DETECTORS:                                          │
│   ├── CenterNet — predict object centers + size                   │
│   ├── FCOS — fully convolutional, per-pixel prediction            │
│   └── DETR — Transformer-based, set prediction                    │
└──────────────────────────────────────────────────────────────────┘
```

---

## **5.2 The R-CNN Family**

```
┌────────────────────────────────────────────────────────────────────────┐
│                    R-CNN FAMILY EVOLUTION                                │
│                                                                        │
│  R-CNN (2014):                                                         │
│    Image → Selective Search (~2000 proposals) → Warp each to 227×227  │
│    → CNN feature extraction per proposal → SVM classifier + regressor │
│    ⚠️ Very slow: ~47s per image (separate CNN per proposal)            │
│                                                                        │
│  Fast R-CNN (2015):                                                    │
│    Image → CNN → Feature Map (shared!) → RoI Pooling per proposal     │
│    → FC layers → Class + Bbox (multi-task loss)                        │
│    ✓ Shared features: ~2s per image (146× faster than R-CNN)          │
│    ⚠️ Still uses external selective search for proposals               │
│                                                                        │
│  Faster R-CNN (2015):                                                  │
│    Image → CNN backbone → Feature Map                                  │
│    ├── Region Proposal Network (RPN) → proposals (learned!)           │
│    └── RoI Pooling → FC → Class + Bbox                                │
│    ✓ End-to-end trainable: ~0.2s per image (5 FPS)                    │
│    ✓ RPN replaces selective search — anchor-based proposals           │
│                                                                        │
│  Mask R-CNN (2017):                                                    │
│    Faster R-CNN + parallel mask prediction branch                      │
│    ├── Classification head → Class label                               │
│    ├── Bounding box head → Bbox regression                            │
│    └── Mask head → Binary mask per RoI (instance segmentation)        │
│    ✓ RoIAlign replaces RoI Pooling (no quantization)                  │
│    ✓ Multi-task: detect + segment simultaneously                      │
└────────────────────────────────────────────────────────────────────────┘
```

### **Region Proposal Network (RPN) — The Key Innovation of Faster R-CNN**

```
┌────────────────────────────────────────────────────────────────┐
│                REGION PROPOSAL NETWORK (RPN)                    │
│                                                                 │
│   Feature Map (from backbone CNN)                               │
│        │                                                        │
│        ▼                                                        │
│   [3×3 Conv sliding window]                                     │
│        │                                                        │
│        ├──→ [1×1 Conv: 2k scores]   objectness (fg/bg)         │
│        │     k anchors × 2 classes                              │
│        │                                                        │
│        └──→ [1×1 Conv: 4k coords]   bbox regression            │
│              k anchors × 4 (dx, dy, dw, dh)                    │
│                                                                 │
│   Anchors at each location:                                     │
│     k = 9 (3 scales × 3 aspect ratios)                          │
│     Scales: {128², 256², 512²}                                  │
│     Ratios: {1:1, 1:2, 2:1}                                    │
│                                                                 │
│   Total anchors for 40×60 feature map:                          │
│     40 × 60 × 9 = 21,600 proposals (before NMS)                │
└────────────────────────────────────────────────────────────────┘
```

---

## **5.3 YOLO Family — You Only Look Once**

YOLO treats detection as a **single regression problem** — predicting bounding boxes and class probabilities directly from the full image in one forward pass.

### **YOLO v1 (2016)**

Divides image into S×S grid. Each cell predicts B bounding boxes + confidence + C class probabilities.

$$
\text{Output tensor: } S \times S \times (B \times 5 + C)
$$

For S=7, B=2, C=20 (PASCAL VOC): 7×7×30 = 1,470 predictions in one pass.

### **YOLO Evolution**

| Version | Year | Key Changes | Speed (FPS) | mAP (COCO) |
|---------|------|-------------|-------------|-------------|
| **v1** | 2016 | Grid-based regression | 45 | 63.4 (VOC) |
| **v2** | 2017 | Batch norm, anchor boxes, multi-scale | 67 | 48.1 |
| **v3** | 2018 | FPN, multi-scale detection (3 scales), Darknet-53 | 30 | 55.3 |
| **v4** | 2020 | CSPDarknet, Mosaic augmentation, CIoU loss | 65 | 65.7 |
| **v5** | 2020 | PyTorch (Ultralytics), auto-anchor, export formats | 140 | 68.9 |
| **v7** | 2022 | E-ELAN, compound scaling, reparameterization | 120+ | 73.0 |
| **v8** | 2023 | Anchor-free, decoupled head, C2f module | 160+ | 73.8 |

### **SSD (Single Shot MultiBox Detector)**

Uses multi-scale feature maps from different layers of the backbone to detect objects at different sizes:

```
┌──────────────────────────────────────────────────────────────────┐
│                  SSD MULTI-SCALE DETECTION                        │
│                                                                   │
│   VGG-16 backbone → progressively smaller feature maps:           │
│                                                                   │
│   38×38 → detects small objects    (many boxes per cell)         │
│   19×19 → detects medium objects                                  │
│   10×10 → detects medium-large objects                            │
│    5×5  → detects large objects                                   │
│    3×3  → detects very large objects                              │
│    1×1  → detects full-image objects                              │
│                                                                   │
│   Each feature map predicts: (c + 4) × k  per cell              │
│   where c = classes, 4 = bbox coords, k = default boxes          │
└──────────────────────────────────────────────────────────────────┘
```

---

## **5.4 Anchor-Based vs Anchor-Free**

| Aspect | Anchor-Based | Anchor-Free |
|--------|-------------|-------------|
| **Examples** | Faster R-CNN, YOLO v3–v5, SSD | CenterNet, FCOS, YOLO v8, DETR |
| **Approach** | Predefined boxes of various sizes/ratios | Predict center points + offsets directly |
| **Hyperparameters** | Anchor scales, ratios, number per cell | Fewer (no anchor design needed) |
| **Training** | Assign GT to anchors by IoU matching | Assign GT to feature map locations |
| **Pros** | Stable training, well-understood | Simpler, fewer hyperparameters |
| **Cons** | Anchor design is dataset-specific | May struggle with extreme aspect ratios |

---

## **5.5 Non-Maximum Suppression (NMS)**

After detection, many overlapping boxes may predict the same object. NMS removes redundant detections:

```
┌───────────────────────────────────────────────────────────────────┐
│                     NMS ALGORITHM                                  │
│                                                                    │
│   Input: list of boxes B with confidence scores S                  │
│   Output: filtered list D                                          │
│                                                                    │
│   1. Sort B by confidence score (descending)                       │
│   2. Pick box with highest score → add to D                        │
│   3. Compute IoU of this box with all remaining boxes              │
│   4. Remove boxes with IoU > threshold (e.g., 0.5)                │
│   5. Repeat from step 2 until B is empty                           │
│                                                                    │
│   Variants:                                                        │
│   ├── Soft-NMS: decay scores instead of hard removal               │
│   │     score_i = score_i · exp(-IoU²/σ)                          │
│   ├── DIoU-NMS: considers center distance, not just IoU            │
│   └── Weighted NMS: merge overlapping boxes (weighted avg)         │
└───────────────────────────────────────────────────────────────────┘
```

---

## **5.6 Detection Metrics**

### **Intersection over Union (IoU)**

$$
\text{IoU} = \frac{|B_{\text{pred}} \cap B_{\text{gt}}|}{|B_{\text{pred}} \cup B_{\text{gt}}|}
$$

```
┌──────────────────────────────────────────────┐
│            IoU VISUALIZATION                  │
│                                               │
│   ┌─────────────┐                             │
│   │  Predicted   │                            │
│   │    ┌────────┼───────┐                     │
│   │    │ Inter- │       │                     │
│   │    │ section│       │                      │
│   └────┼────────┘       │                     │
│        │   Ground Truth │                     │
│        └────────────────┘                     │
│                                               │
│   IoU = Intersection / Union                  │
│   IoU = 0.0  → no overlap                    │
│   IoU = 0.5  → typical match threshold        │
│   IoU = 1.0  → perfect alignment              │
└──────────────────────────────────────────────┘
```

### **IoU Variants for Loss Functions**

| Variant | Formula | Advantage |
|---------|---------|-----------|
| **GIoU** | IoU − (area of enclosing box − union) / area of enclosing box | Gradient even when no overlap |
| **DIoU** | IoU − (center distance² / diagonal of enclosing box²) | Faster convergence |
| **CIoU** | DIoU − α·v (aspect ratio penalty) | Best overall (used in YOLO v4+) |

### **Mean Average Precision (mAP)**

```
┌──────────────────────────────────────────────────────────────────────┐
│                        mAP CALCULATION                                │
│                                                                       │
│   For each class:                                                     │
│     1. Rank all detections by confidence score                        │
│     2. At each detection, compute precision and recall                │
│     3. Plot Precision-Recall curve                                    │
│     4. AP = Area under P-R curve (11-point or all-point interp.)     │
│                                                                       │
│   mAP = mean of AP across all classes                                 │
│                                                                       │
│   COCO Metrics (the standard benchmark):                              │
│   ┌────────────────────┬──────────────────────────────────────────┐  │
│   │ mAP@0.5            │ AP at IoU threshold = 0.5 (PASCAL VOC)  │  │
│   │ mAP@0.75           │ AP at IoU threshold = 0.75 (strict)     │  │
│   │ mAP@[.5:.95]       │ Average AP at IoU from 0.5 to 0.95     │  │
│   │                    │ (step 0.05) — THE primary COCO metric   │  │
│   │ AP_small           │ AP for objects < 32² pixels             │  │
│   │ AP_medium          │ AP for 32² < area < 96²                 │  │
│   │ AP_large           │ AP for objects > 96²                    │  │
│   └────────────────────┴──────────────────────────────────────────┘  │
│                                                                       │
│   Precision = TP / (TP + FP)   "Of all detections, how many correct?"│
│   Recall    = TP / (TP + FN)   "Of all objects, how many detected?"  │
└──────────────────────────────────────────────────────────────────────┘
```

---

# **6. Semantic & Instance Segmentation**

---

## **6.1 Segmentation Task Types**

```
┌──────────────────────────────────────────────────────────────────────┐
│                    SEGMENTATION TAXONOMY                               │
│                                                                       │
│   SEMANTIC SEGMENTATION:                                              │
│     Assign a class label to EVERY pixel                               │
│     No distinction between different instances of same class          │
│     "All person pixels are class 'person'"                            │
│                                                                       │
│   INSTANCE SEGMENTATION:                                              │
│     Detect + segment each object instance separately                  │
│     "Person 1, Person 2, Person 3 are different"                      │
│                                                                       │
│   PANOPTIC SEGMENTATION:                                              │
│     Semantic + Instance combined                                      │
│     Stuff classes (sky, road) + Thing classes (car, person)           │
│     Every pixel gets a label AND instance ID                          │
│                                                                       │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                │
│   │ Original    │  │ Semantic    │  │ Instance    │                │
│   │  [person]   │  │ [■■■ red]  │  │ [■ red-1]  │                │
│   │  [person]   │  │ [■■■ red]  │  │ [■ red-2]  │                │
│   │  [car]      │  │ [■■ blue]  │  │ [■ blue-1] │                │
│   │  [sky]      │  │ [■■ cyan]  │  │ [■■ cyan]  │                │
│   └─────────────┘  └─────────────┘  └─────────────┘                │
└──────────────────────────────────────────────────────────────────────┘
```

---

## **6.2 Key Architectures**

### **FCN — Fully Convolutional Network (2015)**

The pioneering work that replaced FC layers with convolutions for dense prediction:

```
┌──────────────────────────────────────────────────────────────────┐
│                    FCN ARCHITECTURE                                │
│                                                                   │
│  Input Image → CNN Encoder (VGG/ResNet backbone)                  │
│    │                                                              │
│    ▼ (progressively downsampled feature maps)                     │
│  1/2 → 1/4 → 1/8 → 1/16 → 1/32                                 │
│    │                                                              │
│    ▼ (transposed convolution / upsampling)                        │
│  1×1 Conv → predictions at 1/32 scale                            │
│    │                                                              │
│    ▼                                                              │
│  Upsample + skip connections from earlier layers                  │
│    │                                                              │
│    ▼                                                              │
│  Per-pixel class prediction (same size as input)                  │
│                                                                   │
│  Variants:                                                        │
│    FCN-32s: single 32× upsample (coarse)                         │
│    FCN-16s: fuse pool4 + 2× upsample of FCN-32s                 │
│    FCN-8s:  fuse pool3 + 2× upsample of FCN-16s (finest)        │
└──────────────────────────────────────────────────────────────────┘
```

### **U-Net (2015) — Encoder-Decoder with Skip Connections**

Originally for biomedical image segmentation, now the backbone of Stable Diffusion and many medical imaging pipelines:

```
┌──────────────────────────────────────────────────────────────────────┐
│                         U-NET ARCHITECTURE                            │
│                                                                       │
│  Encoder (Contracting)              Decoder (Expanding)               │
│                                                                       │
│  [Input: 572×572×1]                [Output: 388×388×2]               │
│        │                                    ▲                         │
│    [Conv 3×3 ×2] → 64                      │                         │
│        │                            [Conv 3×3 ×2] → 64               │
│    [MaxPool 2×2]                           ▲                         │
│        │              ──skip──→            │                          │
│    [Conv 3×3 ×2] → 128            [UpConv 2×2] + concat              │
│        │                                   ▲                         │
│    [MaxPool 2×2]                          │                          │
│        │              ──skip──→           │                          │
│    [Conv 3×3 ×2] → 256            [UpConv 2×2] + concat              │
│        │                                   ▲                         │
│    [MaxPool 2×2]                          │                          │
│        │              ──skip──→           │                          │
│    [Conv 3×3 ×2] → 512            [UpConv 2×2] + concat              │
│        │                                   ▲                         │
│    [MaxPool 2×2]                          │                          │
│        │                                  │                          │
│    [Conv 3×3 ×2] → 1024 ─── bottleneck ──┘                          │
│                                                                       │
│   Key Innovation: Skip connections from encoder to decoder            │
│   concatenate high-resolution features with upsampled features,       │
│   allowing precise localization while using deep semantics.           │
└──────────────────────────────────────────────────────────────────────┘
```

### **DeepLab Family**

| Version | Year | Key Innovation |
|---------|------|---------------|
| **DeepLab v1** | 2015 | Atrous (dilated) convolution to increase receptive field |
| **DeepLab v2** | 2017 | Atrous Spatial Pyramid Pooling (ASPP) — multi-scale features |
| **DeepLab v3** | 2017 | Improved ASPP with global average pooling, no CRF post-processing |
| **DeepLab v3+** | 2018 | Encoder-decoder with ASPP, Xception backbone |

**Atrous (Dilated) Convolution:**

$$
y[i] = \sum_{k} x[i + r \cdot k] \cdot w[k]
$$

where $r$ is the dilation rate. Increases receptive field without losing resolution or adding parameters.

```
Standard Conv (rate=1):    Dilated Conv (rate=2):    Dilated Conv (rate=4):
  ■ ■ ■                     ■ ○ ■ ○ ■                 ■ ○ ○ ○ ■ ○ ○ ○ ■
  ■ ■ ■                     ○ ○ ○ ○ ○                 ○ ○ ○ ○ ○ ○ ○ ○ ○
  ■ ■ ■                     ■ ○ ■ ○ ■                 ○ ○ ○ ○ ○ ○ ○ ○ ○
  RF: 3×3                   ○ ○ ○ ○ ○                 ○ ○ ○ ○ ○ ○ ○ ○ ○
                             ■ ○ ■ ○ ■                 ■ ○ ○ ○ ■ ○ ○ ○ ■
                             RF: 5×5                   RF: 9×9
```

### **Segment Anything Model (SAM) — Meta, 2023**

A **foundation model** for segmentation that can segment any object given a prompt (point, box, mask, or text):

```
┌──────────────────────────────────────────────────────────────────┐
│                   SAM ARCHITECTURE                                │
│                                                                   │
│   Image Encoder (ViT-H):                                         │
│     Image → [Patch Embeddings] → [ViT-H (632M params)] →        │
│     Image Embedding (256-D per spatial location)                  │
│     ⚠️ Run once per image (amortized)                            │
│                                                                   │
│   Prompt Encoder:                                                 │
│     Points → positional embeddings + learned type embeddings      │
│     Boxes → corner point embeddings                               │
│     Masks → conv embeddings + element-wise sum                    │
│     Text → CLIP text encoder                                      │
│                                                                   │
│   Mask Decoder (lightweight):                                     │
│     [Image Embedding] + [Prompt Tokens]                           │
│     → Transformer decoder (2 layers) → MLP                       │
│     → Predicted mask(s) + IoU confidence score                    │
│                                                                   │
│   Key Properties:                                                 │
│     ✓ Zero-shot segmentation (never seen the object class)       │
│     ✓ Promptable — interactive segmentation                       │
│     ✓ Trained on SA-1B: 11M images, 1.1B masks                  │
│     ✓ Ambiguity-aware: outputs multiple valid masks              │
└──────────────────────────────────────────────────────────────────┘
```

---

# **7. Pose Estimation**

---

## **7.1 Overview**

Pose estimation predicts the spatial locations of body parts or keypoints. It is critical for action recognition, AR/VR, and — in your project — **fashion garment segmentation**.

```
┌──────────────────────────────────────────────────────────────────┐
│                   POSE ESTIMATION TAXONOMY                        │
│                                                                   │
│   TOP-DOWN:                                                       │
│     1. Detect person with object detector                         │
│     2. Estimate keypoints within each person crop                 │
│     ✓ Higher accuracy per instance                                │
│     ✗ Speed depends on number of people                           │
│     Examples: SimpleBaseline, HRNet                               │
│                                                                   │
│   BOTTOM-UP:                                                      │
│     1. Detect ALL keypoints in the image                          │
│     2. Group keypoints into person instances                      │
│     ✓ Speed independent of number of people                      │
│     ✗ Harder to associate keypoints correctly                     │
│     Examples: OpenPose, HigherHRNet                               │
│                                                                   │
│   DENSE POSE (continuous surface):                                │
│     Predicts UV coordinates on a 3D body surface model            │
│     Maps every pixel on a person to a body part + UV location     │
│     ✓ Much richer than keypoints — full surface mapping           │
│     Used in: YOUR fashion recommendation project                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## **7.2 OpenPose**

A **bottom-up** multi-person pose estimation system using two branches:

```
┌────────────────────────────────────────────────────────────────────┐
│                    OPENPOSE ARCHITECTURE                            │
│                                                                     │
│   Input Image                                                       │
│       │                                                             │
│       ▼                                                             │
│   VGG-19 (first 10 layers) → Shared Feature Map                    │
│       │                                                             │
│       ├──→ Branch 1: Confidence Maps (Part Detection)               │
│       │     Heatmap for each keypoint (nose, elbow, etc.)          │
│       │     S = {S₁, S₂, ..., S_J}  for J body parts              │
│       │                                                             │
│       └──→ Branch 2: Part Affinity Fields (PAFs)                   │
│             Vector fields encoding limb associations               │
│             L = {L₁, L₂, ..., L_C}  for C limbs                   │
│                                                                     │
│   Stage refinement (6 stages) improves both branches iteratively   │
│                                                                     │
│   Assembly:                                                         │
│     1. Detect keypoint candidates from confidence maps             │
│     2. Use PAFs to associate keypoints into person skeletons       │
│     3. Hungarian algorithm for optimal bipartite matching          │
│                                                                     │
│   Keypoints (COCO 17): nose, eyes, ears, shoulders, elbows,        │
│     wrists, hips, knees, ankles                                     │
└────────────────────────────────────────────────────────────────────┘
```

---

## **7.3 DensePose — Your Fashion Recommendation Foundation**

DensePose (Facebook AI, 2018) goes far beyond sparse keypoints — it establishes a **dense correspondence** between every pixel in the image and a 3D surface model of the human body.

```
┌────────────────────────────────────────────────────────────────────────┐
│                      DENSEPOSE ARCHITECTURE                             │
│                                                                        │
│   Built on top of Mask R-CNN:                                          │
│                                                                        │
│   Image → ResNet-FPN backbone → Feature Maps                           │
│     │                                                                  │
│     ├── Detection Head → Person bounding boxes                         │
│     │                                                                  │
│     └── DensePose Head (per detected person):                          │
│           │                                                            │
│           ├── Part segmentation → which of 24 body parts               │
│           │     (head, torso, upper arms, lower arms, hands,           │
│           │      upper legs, lower legs, feet — left/right)            │
│           │                                                            │
│           └── UV regression → continuous surface coordinates           │
│                 U ∈ [0, 1] and V ∈ [0, 1] on each body part          │
│                 Maps pixel to 3D SMPL body surface location            │
│                                                                        │
│   Output per pixel on a person:                                        │
│     (body_part_id, U, V) → unique 3D surface location                 │
│                                                                        │
│   24 body surface parts:                                               │
│     ┌─────┐                                                            │
│     │Head │  = part 1,2 (front/back)                                   │
│     ├─────┤                                                            │
│     │Torso│  = part 3,4 (front/back)                                   │
│     ├──┬──┤                                                            │
│     │UA│UA│  = parts 5–8 (upper arms L/R)                              │
│     ├──┼──┤                                                            │
│     │LA│LA│  = parts 9–12 (lower arms L/R)                             │
│     ├──┼──┤                                                            │
│     │UL│UL│  = parts 13–16 (upper legs L/R)                            │
│     ├──┼──┤                                                            │
│     │LL│LL│  = parts 17–20 (lower legs L/R)                            │
│     ├──┼──┤                                                            │
│     │Ft│Ft│  = parts 21–24 (feet L/R)                                  │
│     └──┴──┘                                                            │
│                                                                        │
│   🎯 YOUR PROJECT USAGE:                                               │
│     DensePose segments clothing regions on fashion images,             │
│     enabling masked CLIP embeddings that focus on the garment          │
│     rather than background noise — yielding more accurate              │
│     fashion similarity search.                                         │
└────────────────────────────────────────────────────────────────────────┘
```

### **DensePose in Your Fashion Pipeline**

```python
# From your retrieval.py — applying DensePose segmentation masks
def apply_mask(img_path, mask_path):
    """Apply DensePose segmentation mask to isolate garment region."""
    img = Image.open(img_path).convert('RGB')
    mask = Image.open(mask_path).convert('L')  # Grayscale mask
    # The mask highlights the clothing region from DensePose output
    # This focuses CLIP embedding on the garment, not background
    img_array = np.array(img)
    mask_array = np.array(mask)
    masked = img_array * (mask_array[:, :, None] > 0)
    return Image.fromarray(masked.astype(np.uint8))

# In build_index(): each fashion image is masked before embedding
if os.path.exists(mask_path):
    img = apply_mask(img_path, mask_path)  # DensePose-masked garment
else:
    img = Image.open(img_path).convert('RGB')  # Fallback: full image
```

**Interview Tip:** "We used DensePose to generate per-pixel body part segmentation maps for 44,096 fashion images. By masking out background and non-clothing regions before computing CLIP embeddings, our retrieval accuracy improved significantly — the model focused on garment features like texture, pattern, and cut rather than background clutter."

---

# **8. Vision Transformers**

---

## **8.1 ViT — Vision Transformer (Dosovitskiy et al., 2020)**

The breakthrough paper that showed pure Transformers can match or exceed CNNs on image classification when trained with sufficient data.

```
┌──────────────────────────────────────────────────────────────────────┐
│                    ViT — VISION TRANSFORMER                           │
│                                                                       │
│   Step 1: Patch Embedding                                             │
│     Image (224×224×3) → split into 16×16 patches                     │
│     N = (224/16)² = 196 patches                                      │
│     Each patch: 16×16×3 = 768 pixels → flatten → linear → D-dim     │
│                                                                       │
│   ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐                      │
│   │p1│p2│p3│p4│p5│p6│p7│p8│p9│..│..│..│..│pN│  196 patches           │
│   └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘                      │
│                  │                                                    │
│   Step 2: Prepend [CLS] token + Add Positional Embeddings            │
│     Tokens: [CLS, p₁, p₂, ..., p_N] → (197 × D)                    │
│     z₀ = [x_class; x¹_p·E; x²_p·E; ...] + E_pos                   │
│                  │                                                    │
│   Step 3: Transformer Encoder (L layers)                              │
│     Each layer: LayerNorm → MSA → Residual → LayerNorm → MLP → Res  │
│                  │                                                    │
│   Step 4: Classification Head                                         │
│     Use [CLS] token output → LayerNorm → MLP → class prediction     │
│                                                                       │
│   Architecture Variants:                                              │
│   ┌──────────┬───────┬────────┬───────┬──────────┐                   │
│   │ Model    │ Layers│ Hidden │ Heads │ Params   │                   │
│   ├──────────┼───────┼────────┼───────┼──────────┤                   │
│   │ ViT-B/16 │  12   │  768   │  12   │   86M    │                   │
│   │ ViT-L/16 │  24   │ 1024   │  16   │  307M    │                   │
│   │ ViT-H/14 │  32   │ 1280   │  16   │  632M    │                   │
│   └──────────┴───────┴────────┴───────┴──────────┘                   │
│                                                                       │
│   Key difference from CNNs:                                           │
│   - CNNs have built-in inductive biases: locality, translation eq.   │
│   - ViTs learn spatial relationships from data (need more data)      │
│   - ViTs excel with large-scale pretraining (JFT-300M, ImageNet-21K)│
└──────────────────────────────────────────────────────────────────────┘
```

### **Patch Embedding in Detail**

$$
\mathbf{z}_0 = [\mathbf{x}_{\text{class}}; \; \mathbf{x}_p^1 \mathbf{E}; \; \mathbf{x}_p^2 \mathbf{E}; \; \ldots; \; \mathbf{x}_p^N \mathbf{E}] + \mathbf{E}_{\text{pos}}
$$

where $\mathbf{E} \in \mathbb{R}^{(P^2 \cdot C) \times D}$ is the patch embedding matrix and $\mathbf{E}_{\text{pos}} \in \mathbb{R}^{(N+1) \times D}$ are learned positional embeddings.

---

## **8.2 DeiT — Data-efficient Image Transformers (2021)**

Training ViT requires huge datasets (ImageNet-21K or JFT-300M). DeiT makes ViT practical with **ImageNet-1K only**:

| Innovation | Description |
|-----------|-------------|
| **Knowledge Distillation Token** | Extra learnable `[DIST]` token trained to match teacher's output |
| **Teacher Model** | RegNetY-16GF (CNN) — teaches spatial inductive biases to ViT |
| **Strong Augmentation** | RandAugment, Mixup, CutMix, Random Erasing, Repeated Augmentation |
| **Regularization** | Stochastic depth, Label smoothing |
| **Result** | ViT-B/16 trained on ImageNet-1K → 83.1% top-1 (vs 77.9% ViT without these) |

---

## **8.3 Swin Transformer — Hierarchical Vision Transformer (2021)**

Swin addresses ViT's limitations: quadratic complexity with image size and lack of multi-scale features.

```
┌──────────────────────────────────────────────────────────────────────┐
│                    SWIN TRANSFORMER                                    │
│                                                                       │
│   Key Innovation: SHIFTED WINDOWS                                     │
│                                                                       │
│   Stage 1 (4×4 patches):                                              │
│     ┌──┬──┬──┬──┐    Window-based self-attention:                     │
│     │W1│W1│W2│W2│    Attention computed WITHIN each window             │
│     │W1│W1│W2│W2│    Complexity: O(N × M²) instead of O(N²)          │
│     │W3│W3│W4│W4│    where M = window size (e.g., 7×7 = 49 tokens)   │
│     │W3│W3│W4│W4│                                                     │
│     └──┴──┴──┴──┘                                                     │
│                                                                       │
│   Shifted window (alternating layers):                                │
│     ┌──┬──┬──┬──┐                                                     │
│     │W1│W2│W2│W1│    Shift by (M/2, M/2) pixels                      │
│     │W3│W4│W4│W3│    Creates cross-window connections                 │
│     │W3│W4│W4│W3│    without increasing complexity                    │
│     │W1│W2│W2│W1│                                                     │
│     └──┴──┴──┴──┘                                                     │
│                                                                       │
│   Hierarchical Feature Maps (like CNN):                               │
│     Stage 1: H/4 × W/4 × C        (patch merging)                    │
│     Stage 2: H/8 × W/8 × 2C       (patch merging)                    │
│     Stage 3: H/16 × W/16 × 4C     (patch merging)                    │
│     Stage 4: H/32 × W/32 × 8C                                        │
│                                                                       │
│   ✓ Linear complexity O(N) with image size                            │
│   ✓ Multi-scale features → works as backbone for detection/segm.     │
│   ✓ Replaced ResNet in Mask R-CNN, Cascade R-CNN, etc.               │
└──────────────────────────────────────────────────────────────────────┘
```

### **ViT vs Swin Comparison**

| Aspect | ViT | Swin Transformer |
|--------|-----|-----------------|
| **Attention scope** | Global (all tokens) | Local (within windows) |
| **Complexity** | O(N²) quadratic | O(N) linear |
| **Feature maps** | Single scale | Multi-scale (hierarchical) |
| **Positional encoding** | Absolute | Relative (within windows) |
| **Downstream tasks** | Classification primarily | Classification + Detection + Segmentation |
| **Params (Base)** | 86M | 88M |
| **ImageNet Top-1** | 77.9% (no distill) | 83.5% |

---

# **9. Multimodal Vision-Language Models**

---

## **9.1 CLIP — Contrastive Language-Image Pre-training (Your Core Model)**

CLIP (OpenAI, 2021) learns visual concepts from natural language supervision using **contrastive learning** on 400M image-text pairs.

```
┌──────────────────────────────────────────────────────────────────────┐
│                    CLIP ARCHITECTURE                                   │
│                                                                       │
│   Training: Contrastive Image-Text Matching                           │
│                                                                       │
│   Batch of N (image, text) pairs:                                     │
│                                                                       │
│   Image Encoder             Text Encoder                              │
│   (ViT-B/32 or RN50)       (Transformer)                             │
│       │                         │                                     │
│       ▼                         ▼                                     │
│   [I₁] ─── 512-D          [T₁] ─── 512-D                            │
│   [I₂] ─── 512-D          [T₂] ─── 512-D                            │
│   [I₃] ─── 512-D          [T₃] ─── 512-D                            │
│    ...                      ...                                       │
│   [Iₙ] ─── 512-D          [Tₙ] ─── 512-D                            │
│                                                                       │
│   Similarity Matrix (N × N):                                          │
│                                                                       │
│        T₁    T₂    T₃   ...  Tₙ                                     │
│   I₁ [ ✓    ✗    ✗   ...  ✗ ]   Maximize diagonal                   │
│   I₂ [ ✗    ✓    ✗   ...  ✗ ]   (matching pairs)                    │
│   I₃ [ ✗    ✗    ✓   ...  ✗ ]   Minimize off-diagonal               │
│   ..                              (non-matching pairs)                │
│   Iₙ [ ✗    ✗    ✗   ...  ✓ ]                                       │
│                                                                       │
│   Loss: Symmetric Cross-Entropy                                       │
│     L = ½ (L_image→text + L_text→image)                              │
│     L_i→t = -log( exp(sim(Iᵢ,Tᵢ)/τ) / Σⱼ exp(sim(Iᵢ,Tⱼ)/τ) )    │
│     τ = learnable temperature parameter                               │
└──────────────────────────────────────────────────────────────────────┘
```

### **CLIP Contrastive Loss — Mathematical Formulation**

Given a batch of $N$ image-text pairs, let $\mathbf{i}_k = f_\theta(\text{image}_k)$ and $\mathbf{t}_k = g_\phi(\text{text}_k)$ be the L2-normalized embeddings:

$$
\mathcal{L}_{\text{image} \to \text{text}} = -\frac{1}{N} \sum_{k=1}^{N} \log \frac{\exp(\mathbf{i}_k \cdot \mathbf{t}_k / \tau)}{\sum_{j=1}^{N} \exp(\mathbf{i}_k \cdot \mathbf{t}_j / \tau)}
$$

$$
\mathcal{L}_{\text{text} \to \text{image}} = -\frac{1}{N} \sum_{k=1}^{N} \log \frac{\exp(\mathbf{t}_k \cdot \mathbf{i}_k / \tau)}{\sum_{j=1}^{N} \exp(\mathbf{t}_k \cdot \mathbf{i}_j / \tau)}
$$

$$
\mathcal{L}_{\text{CLIP}} = \frac{1}{2}(\mathcal{L}_{\text{image} \to \text{text}} + \mathcal{L}_{\text{text} \to \text{image}})
$$

### **Zero-Shot Classification with CLIP**

```
┌──────────────────────────────────────────────────────────────────┐
│                 CLIP ZERO-SHOT CLASSIFICATION                     │
│                                                                   │
│   No training needed on target dataset!                           │
│                                                                   │
│   1. Create text prompts for each class:                          │
│      "a photo of a {class_name}"                                 │
│      → "a photo of a dog"                                        │
│      → "a photo of a cat"                                        │
│      → "a photo of a car"                                        │
│                                                                   │
│   2. Encode prompts with text encoder → T₁, T₂, T₃              │
│                                                                   │
│   3. Encode test image with image encoder → I                     │
│                                                                   │
│   4. Compute similarities: sim(I, Tₖ) for each class             │
│                                                                   │
│   5. Predicted class = argmax_k sim(I, Tₖ)                       │
│                                                                   │
│   Result: 76.2% on ImageNet zero-shot                             │
│   (competitive with supervised ResNet-50!)                        │
└──────────────────────────────────────────────────────────────────┘
```

### **Your CLIP + LoRA Fine-Tuning (Fashion Recommendation)**

```python
# From your finetune_lora.py — LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    r=16,                          # Rank (controls capacity)
    lora_alpha=32,                 # Scaling factor
    lora_dropout=0.1,              # Regularization
    target_modules=[
        "q_proj", "k_proj",       # Attention projections
        "v_proj", "out_proj",
        "c_fc", "c_proj",         # Optional MLP layers
    ],
    bias="none",
)

# Parameter efficiency: 98,304 trainable / 102M total = 0.096%

# Contrastive training loss (from your code)
img_features = peft_model.encode_image(imgs)    # Image branch
txt_features = peft_model.encode_text(txts)      # Text branch
img_features = img_features / img_features.norm(dim=-1, keepdim=True)
txt_features = txt_features / txt_features.norm(dim=-1, keepdim=True)

logits = img_features @ txt_features.T / tau     # Similarity matrix
labels = torch.arange(batch_size, device=device) # Diagonal is positive

loss_i = F.cross_entropy(logits, labels)         # Image → Text
loss_t = F.cross_entropy(logits.T, labels)       # Text → Image
loss = (loss_i + loss_t) / 2                     # Symmetric loss
```

---

## **9.2 BLIP and BLIP-2**

### **BLIP (Bootstrapping Language-Image Pre-training)**

BLIP introduces a unified framework for understanding AND generation tasks:

```
┌──────────────────────────────────────────────────────────────────────┐
│                     BLIP ARCHITECTURE                                 │
│                                                                       │
│   THREE OBJECTIVES (Shared Image Encoder):                            │
│                                                                       │
│   1. Image-Text Contrastive (ITC) — like CLIP                        │
│      Aligns image and text representations                            │
│                                                                       │
│   2. Image-Text Matching (ITM) — binary classification                │
│      "Does this image match this text?" (harder negatives via ITC)    │
│                                                                       │
│   3. Language Modeling (LM) — autoregressive generation               │
│      Generate caption conditioned on image                            │
│                                                                       │
│   Key Innovation: CapFilt (Caption and Filter)                        │
│     Captioner: generates synthetic captions for web images            │
│     Filter: removes noisy image-text pairs                            │
│     → Bootstrap cleaner training data                                 │
│                                                                       │
│   Capabilities:                                                       │
│     ✓ Image captioning                                                │
│     ✓ Visual Question Answering (VQA)                                │
│     ✓ Image-text retrieval                                            │
│     ✓ Zero-shot transfer                                              │
└──────────────────────────────────────────────────────────────────────┘
```

### **BLIP-2 — Bridging the Modality Gap with Q-Former**

```
┌──────────────────────────────────────────────────────────────────────┐
│                    BLIP-2 ARCHITECTURE                                 │
│                                                                       │
│   Frozen Image Encoder          Q-Former          Frozen LLM          │
│   (ViT-g, 1B params)           (trainable)       (OPT/FlanT5)        │
│                                                                       │
│   Image → [ViT-g] → visual   → [Q-Former] → soft visual prompts     │
│           features               │                     │              │
│                                  │ 32 learnable        ▼              │
│                                  │ query tokens   [LLM decoder]      │
│                                  │                     │              │
│                                  │ Cross-attention     ▼              │
│                                  │ with visual     text output        │
│                                  │ features                           │
│                                                                       │
│   Q-Former: Lightweight transformer (188M params)                     │
│     - 32 learnable query tokens attend to frozen visual features     │
│     - Bridges the gap between vision encoder and LLM                  │
│     - Trained with ITC + ITM + LM objectives                          │
│                                                                       │
│   Training Strategy (2 stages):                                       │
│     Stage 1: Vision-language representation learning (Q-Former + ViT) │
│     Stage 2: Vision-to-language generation (Q-Former + LLM)          │
│                                                                       │
│   Key Innovation: Only Q-Former trains — both ViT and LLM frozen     │
│   → Extremely parameter-efficient multimodal learning                 │
└──────────────────────────────────────────────────────────────────────┘
```

### **Your BLIP Usage (Fashion Captioning)**

```python
# From your few_shot_blip.py — BLIP generates fashion-specific captions
# These captions are then used to fine-tune CLIP with LoRA

# Pipeline:
# 1. BLIP generates captions for 44,096 fashion images
# 2. Captions stored in captions.jsonl: {"path": "...", "caption": "..."}
# 3. CLIP fine-tuned on (image, caption) pairs with LoRA
# 4. Fine-tuned CLIP generates better fashion embeddings
```

---

## **9.3 LLaVA — Large Language and Vision Assistant (Your Radiology Project)**

```
┌──────────────────────────────────────────────────────────────────────┐
│                    LLaVA ARCHITECTURE                                  │
│                                                                       │
│   Image → [CLIP ViT-L/14] → visual features (256 × 1024)            │
│                │                                                      │
│                ▼                                                      │
│   [Linear Projection] → visual tokens (same dim as LLM)             │
│                │                                                      │
│                ▼                                                      │
│   [Prepend visual tokens to text tokens]                              │
│                │                                                      │
│                ▼                                                      │
│   [LLM (Vicuna/Mistral)] → generates text response                  │
│                                                                       │
│   Training Strategy:                                                  │
│     Stage 1: Feature alignment                                        │
│       Freeze ViT + LLM, train only projection layer                  │
│       Data: 595K image-text pairs (CC3M filtered)                    │
│                                                                       │
│     Stage 2: Visual instruction tuning                                │
│       Freeze ViT, fine-tune projection + LLM with LoRA              │
│       Data: 158K instruction-following data (GPT-4 generated)        │
│                                                                       │
│   LLaVA-NeXT (v1.6) improvements:                                    │
│     ✓ Dynamic high resolution (multi-patch) — up to 672×672          │
│     ✓ Better visual instruction tuning data                          │
│     ✓ Mistral-7B base (instead of Vicuna)                            │
│     ✓ Improved reasoning and OCR capabilities                        │
│                                                                       │
│   🎯 YOUR RADIOLOGY PROJECT:                                         │
│     Fine-tuned LLaVA-NeXT v1.6-Mistral-7B with LoRA (41.9M params) │
│     on MIMIC-CXR for structured radiology report generation          │
│     Multi-pass: Impression → CheXpert labels → ICD codes            │
└──────────────────────────────────────────────────────────────────────┘
```

---

## **9.4 Cross-Modal Fusion Strategies**

```
┌──────────────────────────────────────────────────────────────────────┐
│              CROSS-MODAL FUSION: SHALLOW vs DEEP                      │
│                                                                       │
│   SHALLOW FUSION (Q-Former style — BLIP-2):                          │
│     Vision and language encoders are FROZEN                           │
│     Small bridge module (Q-Former) learns cross-modal alignment       │
│     ✓ Parameter efficient, preserves pretrained knowledge             │
│     ✗ Limited interaction depth between modalities                    │
│                                                                       │
│     [Frozen ViT] ──→ [Q-Former (188M)] ──→ [Frozen LLM]             │
│                       ↑ trainable ↑                                   │
│                                                                       │
│   DEEP FUSION (LLaVA style):                                         │
│     Visual tokens directly fed into LLM alongside text tokens         │
│     LLM's self-attention attends to both modalities jointly           │
│     ✓ Rich cross-modal interaction throughout all layers              │
│     ✗ Requires fine-tuning the LLM (at least with LoRA)             │
│                                                                       │
│     [ViT] ──→ [Linear Proj] ──→ [Visual + Text tokens] ──→ [LLM]   │
│                                    ↑ joint attention ↑                │
│                                                                       │
│   YOUR APPROACH (Fashion Reco):                                       │
│     [CLIP Encoder] → image embedding (512-D)                         │
│     [CLIP Encoder] → text embedding (512-D)                          │
│     Fusion: query = (1-α) × img_emb + α × txt_emb                   │
│     → Weighted linear interpolation in shared CLIP space             │
│     → Simple but effective for retrieval tasks                        │
│                                                                       │
│   YOUR APPROACH (Radiology):                                          │
│     [LLaVA-NeXT] — deep fusion via visual tokens in Mistral-7B      │
│     → Multi-pass generation for structured JSON output               │
│     → Token biasing for clinical label prediction                    │
└──────────────────────────────────────────────────────────────────────┘
```

---

# **10. Generative Vision Models**

---

## **10.1 Stable Diffusion**

```
┌──────────────────────────────────────────────────────────────────────┐
│                   STABLE DIFFUSION ARCHITECTURE                       │
│                                                                       │
│   Text Prompt → [CLIP Text Encoder] → text embeddings                │
│                                            │                          │
│   Latent Space (not pixel space!):         │ cross-attention          │
│                                            ▼                          │
│   z_T (random noise) → [U-Net with attention] → z_0 (clean latent)  │
│                    ↑                                                  │
│                    │ T denoising steps (reverse diffusion)            │
│                    │ Scheduler: DDPM, DDIM, Euler, DPM++             │
│                                                                       │
│   z_0 → [VAE Decoder] → Generated Image (512×512 or higher)         │
│                                                                       │
│   THREE COMPONENTS:                                                   │
│                                                                       │
│   1. VAE (Variational Autoencoder):                                   │
│      Encoder: Image (512×512×3) → Latent (64×64×4)  [8× compression]│
│      Decoder: Latent (64×64×4) → Image (512×512×3)                   │
│      ✓ Working in latent space = 48× fewer pixels to denoise         │
│                                                                       │
│   2. U-Net (Noise Predictor):                                         │
│      Predicts noise ε(z_t, t, c) at each timestep                   │
│      Contains: ResNet blocks + Self-Attention + Cross-Attention       │
│      Cross-attention: attends to CLIP text embeddings                │
│      ~860M parameters (the main compute)                             │
│                                                                       │
│   3. Text Encoder (CLIP ViT-L/14):                                    │
│      Converts text prompt to 77×768 embeddings                        │
│      Provides conditioning signal via cross-attention in U-Net        │
│                                                                       │
│   Training Objective:                                                  │
│     L = E[||ε - ε_θ(z_t, t, c)||²]                                  │
│     Predict the noise that was added at timestep t                    │
│                                                                       │
│   Inference (Sampling):                                               │
│     Start from pure noise z_T, iteratively denoise:                  │
│     z_{t-1} = f(z_t, ε_θ(z_t, t, c))  for t = T, T-1, ..., 1      │
│     Apply Classifier-Free Guidance (CFG):                             │
│     ε̃ = ε_uncond + w·(ε_cond - ε_uncond)                           │
│     w = guidance scale (typically 7–12)                                │
└──────────────────────────────────────────────────────────────────────┘
```

---

## **10.2 DALL-E (OpenAI)**

| Version | Year | Key Approach |
|---------|------|-------------|
| **DALL-E 1** | 2021 | dVAE (discrete VAE) + autoregressive Transformer |
| **DALL-E 2** | 2022 | CLIP embeddings → diffusion prior → diffusion decoder |
| **DALL-E 3** | 2023 | T5-based captioning → improved text faithfulness |

**DALL-E 2 Pipeline:**

```
Text → [CLIP Text Encoder] → text embedding
         │
         ▼
    [Diffusion Prior] → image embedding (predicted CLIP image embed)
         │
         ▼
    [Diffusion Decoder (unCLIP)] → Generated Image
```

---

## **10.3 ControlNet**

ControlNet adds **spatial conditioning** to Stable Diffusion without modifying the base model:

```
┌──────────────────────────────────────────────────────────────────────┐
│                    CONTROLNET ARCHITECTURE                             │
│                                                                       │
│   Condition Input (e.g., Canny edges, pose, depth map)               │
│       │                                                               │
│       ▼                                                               │
│   [Trainable Copy of U-Net Encoder]                                   │
│       │                                                               │
│       │ zero-convolution connections                                  │
│       ▼                                                               │
│   [Locked SD U-Net] ← receives conditioning signals                  │
│       │                                                               │
│       ▼                                                               │
│   Generated Image (follows spatial structure of condition)            │
│                                                                       │
│   Condition Types:                                                    │
│   ├── Canny edges → preserve edge structure                          │
│   ├── OpenPose → control human pose                                  │
│   ├── Depth map → control scene depth/layout                         │
│   ├── Segmentation map → control region classes                      │
│   ├── Normal map → control surface orientation                       │
│   └── Scribble → rough user sketch                                   │
│                                                                       │
│   Key Innovation: "zero convolution" layers initialized to zero       │
│   → No impact on pretrained model at start of training               │
│   → Gradually learns to inject conditioning signal                    │
└──────────────────────────────────────────────────────────────────────┘
```

---

# **11. Transfer Learning for Vision**

---

## **11.1 Two Paradigms**

```
┌──────────────────────────────────────────────────────────────────────┐
│              TRANSFER LEARNING STRATEGIES                              │
│                                                                       │
│   FEATURE EXTRACTION (Frozen Backbone):                               │
│     Freeze all pretrained layers, train only new classifier head      │
│                                                                       │
│     [Frozen ResNet-50] → features → [New FC Layer] → predictions     │
│                                      ↑ only this trains              │
│                                                                       │
│     ✓ Best when: small dataset, target domain similar to source      │
│     ✓ Fast training, no risk of catastrophic forgetting               │
│     ✗ Limited adaptation to new domain                                │
│                                                                       │
│   FINE-TUNING (Full or Partial):                                      │
│     Unfreeze some/all pretrained layers + train new head              │
│                                                                       │
│     [ResNet: early layers frozen | later layers trainable] → [Head]  │
│      ↑ low-level features fixed    ↑ high-level adapted              │
│                                                                       │
│     ✓ Best when: sufficient data, domain shift from ImageNet         │
│     ✓ Better performance ceiling                                      │
│     ✗ Risk of overfitting, needs careful LR scheduling               │
│                                                                       │
│   PEFT — Parameter-Efficient Fine-Tuning (YOUR APPROACH):            │
│     Freeze pretrained model, add small trainable adapters             │
│                                                                       │
│     [Frozen CLIP] + [LoRA adapters: 0.096% params] → fashion domain  │
│      ↑ 102M frozen    ↑ 98K trainable                                │
│                                                                       │
│     ✓ Best of both worlds: preserves pretrained knowledge +          │
│       adapts to new domain with minimal parameters                    │
│     ✓ Multiple LoRA adapters for different domains                    │
│     Used in: your fashion reco (CLIP+LoRA), radiology (LLaVA+LoRA)  │
└──────────────────────────────────────────────────────────────────────┘
```

---

## **11.2 Which Layers to Freeze?**

### **Layer Freezing Strategy**

```
┌──────────────────────────────────────────────────────────────────────┐
│                  LAYER FREEZING DECISION GUIDE                        │
│                                                                       │
│   CNN layers learn progressively abstract features:                   │
│                                                                       │
│   Layer 1–2: Edges, textures (universal — ALWAYS freeze)             │
│   Layer 3–4: Patterns, parts (mostly universal)                      │
│   Layer 5+:  Object-level, semantic (domain-specific)                │
│   FC/Head:   Task-specific (ALWAYS train)                            │
│                                                                       │
│   Decision Matrix:                                                    │
│   ┌──────────────────┬──────────────┬────────────────────────────┐   │
│   │ Dataset Size     │ Domain Shift │ Strategy                   │   │
│   ├──────────────────┼──────────────┼────────────────────────────┤   │
│   │ Small (<1K)      │ Small        │ Feature extraction only    │   │
│   │ Small (<1K)      │ Large        │ Fine-tune last 1–2 blocks │   │
│   │ Medium (1K–10K)  │ Small        │ Fine-tune last few layers  │   │
│   │ Medium (1K–10K)  │ Large        │ Fine-tune most layers      │   │
│   │ Large (10K+)     │ Any          │ Fine-tune all (low LR)     │   │
│   └──────────────────┴──────────────┴────────────────────────────┘   │
│                                                                       │
│   Learning Rate Strategy:                                             │
│     Discriminative LR: lower LR for early layers, higher for later   │
│     Example: early=1e-5, mid=1e-4, head=1e-3                        │
│                                                                       │
│   YOUR APPROACH (LoRA Fine-Tuning):                                   │
│     All base parameters frozen (102M)                                 │
│     LoRA adapters added to attention projections: q, k, v, out       │
│     Only 98,304 params trained (0.096%)                              │
│     AdamW with LR=1e-4, cosine annealing, gradient clipping          │
└──────────────────────────────────────────────────────────────────────┘
```

---

# **12. Your Projects — Deep Dive**

---

## **12.1 MyWardrobe — Multimodal Fashion Recommendation Engine**

### **System Architecture**

```
┌────────────────────────────────────────────────────────────────────────────┐
│                  MYWARDROBE — COMPLETE PIPELINE                             │
│                                                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  DATA PREPARATION (44,096 fashion images from DeepFashion-1)       │   │
│  │                                                                     │   │
│  │  Raw Images → DensePose Segmentation → Masked Garment Images       │   │
│  │     │              │                        │                       │   │
│  │     ▼              ▼                        ▼                       │   │
│  │  Original     Body Part Maps        Clothing-only images            │   │
│  │  images       (24 UV parts)         (background removed)            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                           │                                                │
│                           ▼                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  CAPTION GENERATION (BLIP Few-Shot)                                 │   │
│  │                                                                     │   │
│  │  Masked Images → [BLIP Model] → Fashion-specific captions          │   │
│  │    "A red floral midi dress with V-neckline and short sleeves"     │   │
│  │                                                                     │   │
│  │  Output: captions.jsonl (44,096 image-caption pairs)                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                           │                                                │
│                           ▼                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  LoRA FINE-TUNING OF CLIP                                           │   │
│  │                                                                     │   │
│  │  Base CLIP (RN50, 102M params)                                     │   │
│  │    + LoRA adapters (r=16, α=32, targeting q,k,v,out,c_fc,c_proj)  │   │
│  │    = 98,304 trainable parameters (0.096%)                          │   │
│  │                                                                     │   │
│  │  Training: Symmetric contrastive loss (τ=0.07)                     │   │
│  │  Optimizer: AdamW (lr=1e-4, wd=1e-2)                              │   │
│  │  Scheduler: Cosine annealing                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                           │                                                │
│                           ▼                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  EMBEDDING GENERATION & INDEXING                                    │   │
│  │                                                                     │   │
│  │  44K masked images → [LoRA CLIP] → 512-D embeddings → L2 normalize│   │
│  │    → np.save("embeddings.npy")                                     │   │
│  │    → FAISS IndexFlatIP (exact inner product search)                │   │
│  │    → Also: IndexIVFFlat (approximate, for scale)                   │   │
│  │    → Also: IndexHNSWFlat (graph-based ANN)                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                           │                                                │
│                           ▼                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  MULTIMODAL SEARCH (Real-time, <250ms)                              │   │
│  │                                                                     │   │
│  │  Query: Image + optional text                                       │   │
│  │    │                                                                │   │
│  │    ├── Image → [LoRA CLIP encode_image] → img_emb (512-D)         │   │
│  │    ├── Text  → process_text_query() → [CLIP encode_text] → txt_emb│   │
│  │    │                                                                │   │
│  │    ▼                                                                │   │
│  │  Fusion: query = (1-α) × img_emb + α × txt_emb                    │   │
│  │    α = 0.0 → pure image search                                     │   │
│  │    α = 0.5 → balanced multimodal (default)                         │   │
│  │    α = 1.0 → pure text search                                      │   │
│  │    │                                                                │   │
│  │    ▼                                                                │   │
│  │  query = L2_normalize(query)                                        │   │
│  │    │                                                                │   │
│  │    ▼                                                                │   │
│  │  FAISS.search(query, top_k=10) → ranked results + similarity scores│   │
│  │    │                                                                │   │
│  │    ▼                                                                │   │
│  │  Filter: min_score threshold (0.3–0.5) → return results            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  FRONTEND & DEPLOYMENT                                              │   │
│  │                                                                     │   │
│  │  Streamlit UI → drag-and-drop upload + text query                  │   │
│  │  AI Stylist → LangChain + Ollama for fashion advice                │   │
│  │  Virtual Wardrobe → Supabase PostgreSQL + Object Storage           │   │
│  │  Backend: PyTorch, FAISS, PIL, Supabase client                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────┘
```

### **Key Technical Decisions & Interview Talking Points**

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **CLIP variant** | RN50 (not ViT-B/32) | Stability on CPU; avoided MPS segfaults |
| **LoRA rank** | r=16, α=32 | Balance: enough capacity for fashion domain without overfitting |
| **LoRA targets** | q,k,v,out + optional MLP (c_fc, c_proj) | Attention layers capture semantic relationships; MLP layers capture feature transformations |
| **Embedding dimension** | 512-D | Native CLIP dimension, no projection needed |
| **FAISS index** | IndexFlatIP (exact) + IVFFlat (approx) | Exact for <50K items; IVF for scale |
| **Fusion method** | Linear interpolation with α | Simple, interpretable, user-controllable |
| **DensePose masking** | Pre-compute masks, apply before embedding | Removes background noise, focuses on garment features |
| **Text processing** | Negative prompt remapping | "no jacket" → "shirt t-shirt top" (CLIP can't negate) |

---

## **12.2 Radiology Report Generation**

### **System Architecture**

```
┌──────────────────────────────────────────────────────────────────────────┐
│                RADIOLOGY REPORT GENERATION PIPELINE                       │
│                                                                          │
│  Model: LLaVA-NeXT v1.6-Mistral-7B (7.28B total, 41.9M trainable)     │
│  Dataset: MIMIC-CXR (4,360 training, 770 validation samples)           │
│  Hardware: NVIDIA A100 (Google Colab Pro)                                │
│                                                                          │
│  Stage A — Image-Only Pipeline:                                          │
│    Chest X-ray → [ViT Vision Encoder] → visual features                │
│      → [Projection Layer] → visual tokens                                │
│      → [Mistral-7B + LoRA] → multi-pass generation:                    │
│          Pass 1: Clinical impression (free text)                         │
│          Pass 2: CheXpert labels × 3-5 votes (12 disease classes, JSON) │
│                                                                          │
│  Stage B — Image + EHR Pipeline:                                         │
│    Chest X-ray + Patient EHR (vitals, labs, devices, chronic conditions)│
│      → Same as Stage A, plus:                                            │
│          Pass 3: ICD-10 codes × 3-5 votes (8 diagnostic codes, JSON)   │
│                                                                          │
│  Multi-Pass Rationale:                                                   │
│    Separate passes for text vs. JSON prevent format corruption           │
│    Voting (3-5 independent generations) improves label stability         │
│    Token biasing boosts positive/negative label prediction accuracy      │
│                                                                          │
│  LoRA Configuration:                                                     │
│    rank=16, alpha=32, dropout=0.05                                       │
│    Target: q_proj, k_proj, v_proj, o_proj + gate_proj, up_proj, down_proj│
│    Precision: bf16 on A100                                               │
│    Effective batch size: 32 (4 per device × 8 gradient accumulation)    │
└──────────────────────────────────────────────────────────────────────────┘
```

### **CV Techniques Used in Radiology**

| Technique | Application in Your Project |
|-----------|---------------------------|
| **Image loading & preprocessing** | PIL for DICOM/JPEG, resize to 336×336 for LLaVA-NeXT |
| **Vision encoder (CLIP ViT)** | Extracts visual features from chest X-rays |
| **Multi-patch processing** | LLaVA-NeXT dynamically splits high-res images into patches |
| **Transfer learning (LoRA)** | Fine-tune on medical domain with 0.58% params (41.9M/7.28B) |
| **Multi-label classification** | CheXpert 12-class and ICD 8-class prediction |
| **Structured generation** | JSON output for clinical labels with voting-based stability |

---

# **13. Common Interview Questions (with Strong Answers)**

---

### **Q1: How does a convolution operation work? What is the receptive field?**

**Answer:** "A convolution slides a learned kernel (e.g., 3×3) over the input feature map, computing element-wise multiplication and summation at each position. The output at position (i,j) is:

$$
\text{out}(i,j) = \sum_m \sum_n \text{input}(i+m, j+n) \cdot \text{kernel}(m,n) + \text{bias}
$$

The receptive field is the region of the input image that affects a particular neuron's output. A 3×3 conv has a 3×3 receptive field. Stacking two 3×3 convs gives a 5×5 effective receptive field — this is VGG's insight: two 3×3 convs are more parameter-efficient than one 5×5 conv (18 vs 25 parameters) while achieving the same receptive field, plus they add an extra non-linearity."

---

### **Q2: Explain the difference between semantic segmentation, instance segmentation, and object detection.**

**Answer:** "Object detection outputs bounding boxes + class labels per object. Semantic segmentation assigns a class label to every pixel but doesn't distinguish between instances — all 'person' pixels share one label. Instance segmentation combines both: it produces per-pixel masks for each individual object instance. Panoptic segmentation unifies all three, labeling every pixel with both a class and an instance ID. In my radiology project, the CheXpert label prediction is essentially multi-label classification on the whole image, while DensePose in my fashion project performs dense pixel-level body part segmentation, which is even richer than instance segmentation — it maps each pixel to a UV coordinate on a 3D body surface."

---

### **Q3: Why do ResNet skip connections help? What problem do they solve?**

**Answer:** "Skip connections solve the degradation problem — deeper networks paradoxically had higher training error than shallower ones, not due to overfitting but because gradients vanished through many layers. The residual connection $y = F(x) + x$ ensures gradients can flow directly through the shortcut during backpropagation. Mathematically, the gradient includes an additive identity term: $\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \cdot (1 + \frac{\partial F}{\partial x})$, meaning even if $\frac{\partial F}{\partial x}$ is small, the gradient doesn't vanish because of the 1. Additionally, the network only needs to learn the residual $F(x) = H(x) - x$, which is easier — if the identity mapping is optimal, driving $F(x)$ to zero is simpler than learning $H(x) = x$ from scratch."

---

### **Q4: How does CLIP's contrastive learning work, and how did you fine-tune it?**

**Answer:** "CLIP aligns image and text embeddings in a shared 512-D space using a symmetric contrastive loss. Given a batch of N image-text pairs, it creates an N×N similarity matrix and uses cross-entropy to maximize similarity for matching pairs (diagonal) while minimizing it for non-matching pairs (off-diagonal). The temperature parameter τ controls the sharpness of the distribution.

In my fashion recommendation project, I fine-tuned CLIP using LoRA — adding low-rank adaptation matrices to the attention layers (q, k, v, out projections). With rank 16 and alpha 32, this gave us only 98,304 trainable parameters out of 102 million (0.096%). We trained on 44,096 fashion image-caption pairs using the same symmetric contrastive loss with τ=0.07. BLIP generated the fashion-specific captions. The LoRA approach was critical because: (1) we preserved CLIP's general visual understanding, (2) we added fashion-specific knowledge with minimal data, and (3) we could swap in different LoRA adapters for different domains."

---

### **Q5: Explain mAP@[.5:.95] and why it's the standard COCO metric.**

**Answer:** "mAP@[.5:.95] averages the mean Average Precision across 10 IoU thresholds from 0.5 to 0.95 in steps of 0.05. At each threshold, a detection is counted as a true positive only if its IoU with the ground truth exceeds that threshold. Then we compute the Precision-Recall curve per class, calculate the area under it (AP), and average across all classes (mAP). Finally, we average across all 10 thresholds.

This metric is more comprehensive than mAP@0.5 alone because: (1) mAP@0.5 is lenient — a box overlapping only 50% with the ground truth counts as correct, (2) mAP@0.75 is stricter and rewards better localization, (3) the averaged metric across all thresholds ensures models can't cheat with loose boxes. It's the primary COCO metric because it balances recognition accuracy with localization precision."

---

### **Q6: What is the difference between anchor-based and anchor-free detection?**

**Answer:** "Anchor-based detectors (Faster R-CNN, YOLO v3-v5, SSD) predefine a set of reference boxes at each feature map location with various scales and aspect ratios. The network predicts offsets from these anchors. This requires careful hyperparameter tuning — choosing the right number, scales, and ratios. Anchor-free detectors (CenterNet, FCOS, YOLO v8) predict object properties directly — either as center points with width/height, or as per-pixel predictions with distance to bounding box edges.

The trend has moved toward anchor-free because: fewer hyperparameters, simpler training pipelines, and competitive accuracy. YOLO v8 switched to anchor-free with a decoupled detection head, and DETR uses a Transformer-based set prediction approach that's entirely anchor-free."

---

### **Q7: How does your DensePose-based masking improve fashion recommendation?**

**Answer:** "DensePose maps every pixel on a detected person to one of 24 body surface parts with continuous UV coordinates. In our fashion pipeline, we use these dense body part predictions to create segmentation masks that isolate the clothing region from the background.

Without masking, CLIP embeddings capture background elements — store shelves, mannequin stands, lighting conditions — which are irrelevant for fashion similarity. With DensePose masks, we extract only the garment pixels before computing CLIP embeddings. This means two identical dresses photographed against different backgrounds will produce much more similar embeddings. The masked images are fed through our LoRA fine-tuned CLIP encoder to generate 512-D vectors that capture garment-specific features: texture, pattern, color, silhouette, and cut. These vectors are then indexed in FAISS for sub-250ms retrieval."

---

### **Q8: Walk me through how Stable Diffusion generates an image from a text prompt.**

**Answer:** "Stable Diffusion works in three stages:

First, the text prompt is encoded by CLIP's text encoder into a sequence of 77 embeddings (77×768). These serve as the conditioning signal.

Second, starting from pure Gaussian noise in the latent space (64×64×4), the U-Net iteratively denoises over T timesteps (typically 20–50 with modern schedulers). At each step, the U-Net predicts the noise component. It uses self-attention for spatial coherence and cross-attention to attend to the text embeddings, ensuring the generated content matches the prompt. Classifier-Free Guidance (CFG) amplifies the text conditioning: the model runs twice — once with the prompt and once without — and the final prediction is an extrapolation toward the conditioned output.

Third, the denoised latent (64×64×4) is decoded by the VAE decoder into a full-resolution image (512×512×3). The VAE's 8× spatial compression is key — denoising happens on a 48× smaller representation than pixel space, making the process tractable.

The key insight is working in latent space rather than pixel space, which is what makes 'Stable' Diffusion practical compared to earlier pixel-space diffusion models."

---

### **Q9: Compare Vision Transformers (ViT) to CNNs. When would you choose each?**

**Answer:** "CNNs have strong inductive biases — locality (nearby pixels are more related) and translation equivariance (features are position-independent). These biases help with small to medium datasets. ViTs lack these biases; they treat an image as a sequence of patches with global self-attention, so they need to learn spatial relationships entirely from data.

I'd choose CNNs when: (1) data is limited (<100K images), (2) speed is critical (MobileNet, EfficientNet for edge), (3) the task needs strong inductive biases (medical imaging with small datasets). I'd choose ViTs when: (1) large-scale pretraining is available (ViT-L with ImageNet-21K), (2) the task benefits from global context (e.g., image-text alignment in CLIP), (3) using a pretrained multimodal model.

In practice, Swin Transformer offers the best of both — hierarchical features like CNNs plus attention's flexibility. In my projects, I used CLIP's vision encoder (either RN50 or ViT-B/32) depending on stability requirements, and LLaVA uses a ViT-L/14 for rich visual representations."

---

### **Q10: How did you handle the medical imaging domain in your radiology project?**

**Answer:** "Medical imaging has unique challenges: limited labeled data, class imbalance (most X-rays are 'No Finding'), and the need for structured output (clinical labels, not just free text).

We addressed these with: (1) LLaVA-NeXT v1.6 as the base — a strong vision-language model pretrained on natural images that we adapted to medical domain, (2) LoRA fine-tuning with 41.9M trainable parameters out of 7.28B total — preserving general visual understanding while adding medical knowledge, (3) curriculum learning — Stage A (image-only, 83.1%) and Stage B (image+EHR, 16.9%) training, (4) multi-pass generation — separate passes for impression text and CheXpert/ICD JSON labels to prevent format corruption, (5) voting-based prediction — 3-5 independent generations per label set, majority vote for stability, (6) token biasing to improve positive/negative label prediction accuracy. The system generates structured reports: clinical impression + 12 CheXpert disease labels + 8 ICD-10 diagnostic codes."

---

# **14. Key Takeaways**

---

```
┌────────────────────────────────────────────────────────────────────────────┐
│                          KEY TAKEAWAYS                                      │
│                                                                            │
│  1. IMAGE FUNDAMENTALS                                                     │
│     Images are H×W×C tensors. Know PyTorch (CHW) vs NumPy (HWC).         │
│     OpenCV loads BGR; convert to RGB for deep learning.                    │
│     ImageNet normalization (mean/std) is the universal standard.           │
│                                                                            │
│  2. AUGMENTATION IS CRITICAL                                               │
│     RandomResizedCrop + Flip + ColorJitter = training baseline.            │
│     Mixup/CutMix regularize, RandAugment automates. Match to task.        │
│                                                                            │
│  3. CLASSICAL CV IS STILL RELEVANT                                         │
│     Canny/Sobel for edges, HSV for color filtering, morphological         │
│     ops for mask cleanup. Interview staple and production workhorse.       │
│                                                                            │
│  4. KNOW YOUR CNN MILESTONES                                               │
│     ResNet (skip connections), EfficientNet (compound scaling),            │
│     MobileNet (depthwise separable). Each solved a specific problem.       │
│                                                                            │
│  5. DETECTION: TWO-STAGE vs ONE-STAGE                                      │
│     Faster R-CNN (RPN + RoI pooling) for accuracy.                        │
│     YOLO family for speed. Trend: anchor-free (v8, DETR).                 │
│     mAP@[.5:.95] is THE metric. Know NMS and IoU variants.               │
│                                                                            │
│  6. SEGMENTATION: ENCODER-DECODER IS THE PATTERN                          │
│     U-Net (skip connections), DeepLab (atrous/ASPP), SAM (promptable).   │
│     DensePose = dense surface correspondence (used in your project).       │
│                                                                            │
│  7. VISION TRANSFORMERS DOMINATE                                           │
│     ViT: patch embeddings + transformer encoder. Simple but data-hungry.  │
│     Swin: shifted windows for efficiency + hierarchical features.          │
│     Know the tradeoffs: inductive bias (CNN) vs flexibility (ViT).        │
│                                                                            │
│  8. MULTIMODAL IS THE FRONTIER                                             │
│     CLIP: contrastive image-text alignment. Zero-shot transfer.           │
│     BLIP-2: Q-Former bridges frozen ViT + frozen LLM (parameter efficient)│
│     LLaVA: visual tokens in LLM (deep fusion). Your radiology project.   │
│                                                                            │
│  9. YOUR DIFFERENTIATORS                                                   │
│     Fashion Reco: CLIP + LoRA (0.096% params) + DensePose + FAISS        │
│       Full pipeline: BLIP captioning → LoRA fine-tuning → masked         │
│       embeddings → multimodal search → <250ms latency on 44K items       │
│     Radiology: LLaVA-NeXT + LoRA → multi-pass structured generation     │
│       Image-only AND Image+EHR pipelines, CheXpert + ICD prediction      │
│                                                                            │
│  10. TRANSFER LEARNING HIERARCHY                                           │
│      Feature extraction (freeze all) → Fine-tune last layers →           │
│      Fine-tune all → PEFT/LoRA (your approach — best of both worlds)     │
│      Know WHEN and WHY to choose each strategy.                           │
└────────────────────────────────────────────────────────────────────────────┘
```

### **30-Second Elevator Pitch**

> "I've built end-to-end computer vision systems across domains. My fashion recommendation engine uses LoRA fine-tuned CLIP with DensePose segmentation and FAISS vector search — achieving sub-250ms retrieval across 44K products with just 0.096% trainable parameters. My radiology system fine-tunes LLaVA-NeXT to generate structured clinical reports from chest X-rays with multi-pass JSON generation and voting-based label prediction. I'm comfortable with classical CV in OpenCV, modern architectures from ResNet to Vision Transformers, and multimodal models like CLIP and LLaVA."

---

*Document prepared for Rahul Sharma — Computer Vision interview preparation. Covers fundamentals through cutting-edge multimodal systems, grounded in hands-on project experience.*
