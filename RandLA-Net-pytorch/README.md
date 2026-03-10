# RandLA-Net — Semantic Segmentation on SemanticKITTI

## Overview

This project implements **RandLA-Net** for **3D semantic segmentation of LiDAR point clouds** on the **SemanticKITTI dataset**.

RandLA-Net is designed to efficiently process **large-scale point clouds** by using random sampling and local feature aggregation instead of expensive point sampling strategies.

Reference paper:

**RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds**

https://arxiv.org/abs/1911.11236

---

# Architecture

RandLA-Net is a **lightweight encoder–decoder architecture** designed for large-scale point clouds.

Main components:

## 1. Input representation

The network processes raw point clouds using:

- point coordinates `(x, y, z)`
- k-nearest neighbors (KNN)

---

## 2. Local Feature Aggregation (LFA)

RandLA-Net extracts local geometric features using:

- relative position encoding
- attention-based pooling
- neighbor feature aggregation

This allows the network to learn spatial relationships between points.

---

## 3. Random Sampling

Instead of expensive **Farthest Point Sampling (FPS)**, RandLA-Net uses **random sampling** to reduce computational cost while preserving performance.

---

## 4. Encoder–Decoder structure

The network follows a hierarchical structure:




---

# Dataset

We use the **SemanticKITTI dataset**, which provides point-wise annotations for LiDAR scans from autonomous driving scenes.

Expected dataset structure:





Official dataset:

https://semantic-kitti.org/

---

# Installation 

## 1. Create environnement and launch some command to trainning

```bash
conda create -n randlanet python=3.9
conda activate randlanet

pip install -r requirements.txt

sh compile_op.sh

sbatch train_randlnet.sbatch



