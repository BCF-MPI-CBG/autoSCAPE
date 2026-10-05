import pandas as pd
import numpy as np
from skimage import measure
from ome_zarr.classes import OMEZarrScene

def object_detection_on_scene(scenes: list[OMEZarrScene], model, resolution_level: int) -> "napari.layers.Shapes":
	import tqdm
	from napari.layers import Shapes
	import os

	shapes_layers = []
	features = pd.DataFrame()

	for scene in scenes:
		images = scene.images
		for name, ms in tqdm.tqdm(images.items()):
			# retrieve translation and multiscales
			translation = [t for t in scene.metadata.coordinateTransformations if t.input.path == name][0]
			translate = translation.translation
			scale=list(ms.images[resolution_level].scale.values())

			projection = ms.images[resolution_level].data.min(axis=0)
			projection = (projection - projection.min()) / (projection.max() - projection.min()) * 255
			z0 = translate[0] + ms.images[resolution_level].data.shape[0] * scale[0] / 2

			df_rectangles = sliding_window_inference(projection.compute(), model, window_size=(640, 640), overlap=0.2)
			df_rectangles = remove_duplicate_detections(df_rectangles, iou_threshold=0.1)
			df_rectangles = extract_features(projection.compute(), df_rectangles)
			
			rectangles = []
			for _, row in df_rectangles.iterrows():
				x0 = float(row['x0']) * scale[2] + translation.translation[2]
				y0 = float(row['y0']) * scale[1] + translation.translation[1]
				x1 = float(row['x1']) * scale[2] + translation.translation[2] 
				y1 = float(row['y1']) * scale[1] + translation.translation[1]
				rectangles.append([(z0, y0, x0), (z0, y0, x1), (z0, y1, x1), (z0, y1, x0)])

			df_rectangles.drop(columns=["x0", "y0", "x1", "y1"], inplace=True)
			df_rectangles["image_name"] = ms.name
			df_rectangles["x"] = (x0 + x1) / 2
			df_rectangles["y"] = (y0 + y1) / 2
		
			features = pd.concat([features, df_rectangles], ignore_index=True)

			shapes_layers.append(Shapes(
					rectangles, shape_type="rectangle", edge_color="red",
					face_color="transparent", features=df_rectangles, edge_width = 0.00001,
					name=os.path.basename(ms.name) + "_detections"
				)
			)

	# concatenate all shapes layers into one
	shapes = [shape for layer in shapes_layers for shape in layer.data]

	return Shapes(shapes, features=features, units=list(ms.images[0].axes_units.values()))


def yolo_box2rect(box):
	y_center, x_center, height, width = box
	x_min = int(x_center - width / 2)
	x_max = int(x_center + width / 2)
	y_min = int(y_center - height / 2)
	y_max = int(y_center + height / 2)

	p0 = (x_min, y_min)
	p1 = (x_max, y_min)
	p2 = (x_max, y_max)
	p3 = (x_min, y_max)

	rect = [p0, p1, p2, p3]
	return rect

def sliding_window_inference(image, model, window_size=(640, 640), overlap=0.2):
	height, width = image.shape[:2]
	win_h, win_w = window_size
	stride_h = int(win_h * (1 - overlap))
	stride_w = int(win_w * (1 - overlap))

	windows = {}

	# Generate all window positions
	y_positions = list(range(0, height - win_h + 1, stride_h))
	x_positions = list(range(0, width - win_w + 1, stride_w))

	# Ensure the far edges are covered by adding a final window at the edge
	if y_positions[-1] + win_h < height:
		y_positions.append(height - win_h)
	if x_positions[-1] + win_w < width:
		x_positions.append(width - win_w)

	# Slide windows across image
	for y_offset in y_positions:
		for x_offset in x_positions:
			window = image[y_offset:y_offset+win_h, x_offset:x_offset+win_w]
			window_norm = (window - window.min()) / (window.max() - window.min() + 1e-8) * 255
			windows[(y_offset, x_offset)] = window_norm

	predictions = model(list(windows.values()), verbose=False)
	df = pd.DataFrame()
	for window, prediction in zip(windows.keys(), predictions):
		y_offset, x_offset = window
		for box, conf in zip(prediction.boxes.xywh, prediction.boxes.conf):
			rectangle = yolo_box2rect(box)
			# Shift rectangle coordinates by window offset
			rectangle_shifted = [(y + y_offset, x + x_offset) for (y, x) in rectangle]
			_df = pd.DataFrame({
				'x0': [rectangle_shifted[0][1]],
				'y0': [rectangle_shifted[0][0]],
				'x1': [rectangle_shifted[2][1]],
				'y1': [rectangle_shifted[2][0]],
				'confidence': [conf.item()]
			})
			df = pd.concat([df, _df], ignore_index=True)

	return df


def extract_features(image: np.ndarray, boxes: pd.DataFrame) -> pd.DataFrame:
	"""Basic feature extraction using skimage.measure.regionprops_table."""

	features = pd.DataFrame()

	for _, box in boxes.iterrows():
		x0, y0, x1, y1 = int(box['x0']), int(box['y0']), int(box['x1']), int(box['y1'])
		crop = image[y0:y1, x0:x1]

		_features = pd.DataFrame(measure.regionprops_table(
			label_image=np.ones(crop.shape, dtype=int),  # Dummy label image
			intensity_image=crop,
			properties=[
				"area",
				"mean_intensity",
				"max_intensity",
				"min_intensity",
				"std_intensity"
			],
		))
		_features["x0"] = x0
		_features["y0"] = y0
		_features["x"] = (x0 + x1) / 2
		_features["x1"] = x1
		_features["y1"] = y1
		_features["y"] = (y0 + y1) / 2
		_features["width"] = x1 - x0
		_features["height"] = y1 - y0
		_features["aspect_ratio"] = _features["width"] / _features["height"]
		if "confidence" in box:
			_features["confidence"] = box["confidence"]
		features = pd.concat([features, _features], ignore_index=True)

	# get distance to nearest other box
	positions = features[["x0", "y0", "x1", "y1"]].values
	for i, box in features.iterrows():
		box_coords = np.array([box['x0'], box['y0'], box['x1'], box['y1']])
		other_boxes = np.delete(positions, i, axis=0)
		distances = np.linalg.norm(other_boxes - box_coords, axis=1)
		features.at[i, "nearest_box_distance"] = distances.min()

	# get area over largest intersection with other boxes
	for i, box in features.iterrows():
		box_coords = np.array([box['x0'], box['y0'], box['x1'], box['y1']])
		boxArea = (box_coords[2] - box_coords[0]) * (box_coords[3] - box_coords[1])
		
		other_boxes = np.delete(positions, i, axis=0)
		IoAs = []
		for other in other_boxes:
			xA = max(box_coords[0], other[0])
			yA = max(box_coords[1], other[1])
			xB = min(box_coords[2], other[2])
			yB = min(box_coords[3], other[3])
			interArea = max(0, xB - xA) * max(0, yB - yA)
			
			if boxArea > 0:
				IoAs.append(interArea / boxArea)
		features.at[i, "max_intersection_over_area"] = max(IoAs) if IoAs else 0

	return features

def remove_duplicate_detections(df, iou_threshold=0.5):
	"""Remove duplicate detections in overlap regions using IoU."""
	from ultralytics.utils.metrics import box_iou
	import torch
	
	if len(df) == 0:
		return df
	
	# Sort by confidence (highest first)
	df = df.sort_values('confidence', ascending=False).reset_index(drop=True)
	keep = []

	boxes_xyxy = df[['x0', 'y0', 'x1', 'y1']].values
	ious = box_iou(torch.tensor(boxes_xyxy), torch.tensor(boxes_xyxy)).numpy()
	np.fill_diagonal(ious, 0)  # Ignore self-comparison
	
	for i, row in df.iterrows():
		keep_this = True
		for kept_idx in keep:
			if ious[i, kept_idx] > iou_threshold:
				keep_this = False
				break
		
		if keep_this:
			keep.append(i)
	
	return df.loc[keep].reset_index(drop=True)