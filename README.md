# DL_Project_Team25
# Project: Data-Driven Feature Tracking for Aerial Imagery

## Objective
This project applies a data-driven feature tracking method, originally designed for event cameras, to aerial imagery. The goal is to extract features and build feature tracks over a sequence of images, using these tracks to estimate 3D camera poses via a Structure-from-Motion (SfM) algorithm. The quality of the recovered camera poses will be evaluated for accuracy and robustness.

## Goals
1. Develop a robust feature tracking pipeline for aerial imagery using event cameras.
2. Build feature tracks from an aerial image sequence and integrate them with an SfM algorithm to estimate 3D camera poses.
3. Evaluate the accuracy and reliability of the reconstructed camera poses.

## Dataset
- **Aerial Imagery**: Provided dataset includes imagery captured using both event cameras and RGB cameras.
- **Preprocessing**: Event data will be converted into frame-like structures for compatibility with feature tracking, and RGB frames will be synchronized with event frames for consistency.

## Project Steps

### Step 1: Feature Extraction from Aerial Imagery
   - Use the data-driven feature tracking method based on Messikommer et al.'s paper, *Data-Driven Feature Tracking for Event Cameras* (CVPR 2023).
   - Adapt the event-based feature tracking approach to work with aerial imagery, leveraging high temporal resolution of event data for improved tracking.

### Step 2: Dataset Preparation
   - **Data Synchronization**: Align event camera data with RGB frames based on timestamps.
   - **Data Preprocessing**: Convert event data into frames and normalize RGB data for consistency across the dataset.

### Step 3: Feature Tracking Across Image Sequence
   - Track features across the entire sequence of aerial images.
   - Use the frame attention module from Messikommer et al.'s method to maintain consistency of feature tracks within each frame.

### Step 4: Generate Feature Tracks for SfM
   - Use the tracked features to create inputs for an SfM algorithm.
   - Ensure that feature tracks are accurate and consistent across the sequence for reliable 3D pose estimation.

### Step 5: 3D Pose Estimation with SfM
   - Apply an SfM algorithm (e.g., COLMAP or BA4S) to estimate 3D camera poses using the feature tracks.
   - Feed the feature tracks into SfM to ensure robustness in the estimated poses.

### Step 6: Evaluation of Results
   - Evaluate the quality of the estimated 3D camera poses.
   - Metrics for evaluation include tracking accuracy, feature track consistency, and robustness across different environmental conditions.

## Installation

### Prerequisites
- Python 3.8+
- Libraries: `OpenCV`, `numpy`, `scipy`, `colmap`
- Clone this repository and install dependencies:

   ```bash
   git clone <repo_url>
   cd project_name
   pip install -r requirements.txt
