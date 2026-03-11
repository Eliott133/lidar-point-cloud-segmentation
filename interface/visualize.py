# import streamlit as st
# import numpy as np
# import pydeck as pdk
# import yaml
# import os
# import pandas as pd

# DATASET_PATH = "/info/corpus/SemanticKITTI/dataset/sequences"
# CONFIG_FILE = "./configs/semantic-kitti.yaml"

# # ------------------------
# # Load config
# # ------------------------

# @st.cache_resource
# def load_config():
#     with open(CONFIG_FILE) as f:
#         return yaml.safe_load(f)

# config = load_config()
# color_map = config["color_map"]


# # ------------------------
# # Load pointcloud
# # ------------------------

# def load_pointcloud(path):

#     points = np.fromfile(path, dtype=np.float32)
#     points = points.reshape((-1,4))

#     return points[:,:3]


# def load_labels(path):

#     labels = np.fromfile(path, dtype=np.uint32)
#     labels = labels & 0xFFFF

#     return labels


# # ------------------------
# # UI
# # ------------------------

# st.title("SemanticKITTI Web Viewer")

# sequences = sorted(os.listdir(DATASET_PATH))
# seq = st.selectbox("Sequence", sequences)

# velodyne = os.path.join(DATASET_PATH, seq, "velodyne")
# labels_dir = os.path.join(DATASET_PATH, seq, "labels")

# frames = sorted(os.listdir(velodyne))

# frame_id = st.slider("Frame", 0, len(frames)-1, 0)

# bin_file = os.path.join(velodyne, frames[frame_id])
# label_file = os.path.join(labels_dir, frames[frame_id].replace(".bin",".label"))

# points = load_pointcloud(bin_file)
# labels = load_labels(label_file)


# # subsample
# max_points = 80000

# if len(points) > max_points:

#     idx = np.random.choice(len(points), max_points, replace=False)

#     points = points[idx]
#     labels = labels[idx]


# # colors
# colors = np.zeros((len(labels),3))

# for l,c in color_map.items():
#     mask = labels == l
#     colors[mask] = c


# df = pd.DataFrame({
#     "x": points[:,0],
#     "y": points[:,1],
#     "z": points[:,2],
#     "r": colors[:,0],
#     "g": colors[:,1],
#     "b": colors[:,2]
# })

# st.write(points.shape)
# st.write(points[:5])


# # ------------------------
# # PyDeck layer
# # ------------------------

# layer = pdk.Layer(
#     "PointCloudLayer",
#     df,
#     get_position=["x", "y", "z"],
#     get_color=["r", "g", "b"],
#     point_size=2
# )

# view_state = pdk.ViewState(
#     target=[0, 0, 0],   # centre de la scène
#     zoom=0,
#     pitch=45,
#     bearing=0
# )

# deck = pdk.Deck(
#     layers=[layer],
#     initial_view_state=view_state,
# )

# st.pydeck_chart(deck)
