import numpy as np
#import api_01
import math
import cv2
import os
import json
#from scipy.special import expit
#from utils.box import BoundBox, box_iou, prob_compare
#from utils.box import prob_compare2, box_intersection
from ...utils.box import BoundBox
from ...cython_utils.cy_yolo2_findboxes import box_constructor

a = dict()
names = []
ds = True
try :
	from deep_sort.application_util import preprocessing as prep
	from deep_sort.application_util import visualization
	from deep_sort.deep_sort.detection import Detection
except :
	ds = False


def expit(x):
	return 1. / (1. + np.exp(-x))

def _softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out

def findboxes(self, net_out):
	# meta
	meta = self.meta
	boxes = list()
	boxes=box_constructor(meta,net_out)
	return boxes


def extract_boxes(self,new_im):
    cont = []
    new_im=new_im.astype(np.uint8)
    ret, thresh=cv2.threshold(new_im, 127, 255, 0)
    p, contours, hierarchy=cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(0, len(contours)):
        cnt=contours[i]
        x, y, w, h=cv2.boundingRect(cnt)
        if w*h > 30**2 and ((w < new_im.shape[0] and h <= new_im.shape[1]) or (w <= new_im.shape[0] and h < new_im.shape[1])):
            if self.FLAGS.tracker == "sort":
                cont.append([x, y, x+w, y+h])
            else : cont.append([x, y, w, h])
    return cont


def postprocess(self, result, net_out, im, frame_id=0, csv_file=None, csv=None, mask=None, encoder=None, tracker=None):
	"""
	Takes net output, draw net_out, save to disk
	"""
	boxes = self.findboxes(net_out)
	# meta
	meta = self.meta
	nms_max_overlap = 0.1
	threshold = meta['thresh']
	colors = meta['colors']
	labels = meta['labels']
	if type(im) is not np.ndarray:
		imgcv = cv2.imread(im)
	else: imgcv = im
	h, w, _ = imgcv.shape
	thick = int((h + w) // 300)
	resultsForJSON = []
	if not self.FLAGS.track :
		for b in boxes:
			boxResults = self.process_box(b, h, w, threshold)
			if boxResults is None:
				continue
			left, right, top, bot, mess, max_indx, confidence = boxResults
			if self.FLAGS.json:
				resultsForJSON.append({"label": mess, "confidence": float('%.2f' % confidence), "topleft": {"x": left, "y": top}, "bottomright": {"x": right, "y": bot}})
				continue
			if self.FLAGS.display or self.FLAGS.saveVideo:
				cv2.rectangle(imgcv,
					(left, top), (right, bot),
					colors[max_indx], thick)
				cv2.putText(imgcv, mess, (left, top - 12),
					0, 1e-3 * h, colors[max_indx],thick//3)
	else :
		if not ds :
			print("ERROR : deep sort or sort submodules not found for tracking please run :")
			print("\tgit submodule update --init --recursive")
			print("ENDING")
			exit(1)
		detections = []
		scores = []
		for b in boxes:
			boxResults = self.process_box(b, h, w, threshold)
			if boxResults is None:
				continue
			left, right, top, bot, mess, max_indx, confidence = boxResults
			if mess not in self.FLAGS.trackObj :
				continue
			if self.FLAGS.tracker == "deep_sort":
				detections.append(np.array([left, top, right-left, bot-top]).astype(np.float64))
				scores.append(confidence)
			elif self.FLAGS.tracker == "sort":
				detections.append(np.array([left, top, right, bot]).astype(np.float64))
		if len(detections) < 3  and self.FLAGS.BK_MOG:
			detections = detections + extract_boxes(self, mask)

		detections = np.array(detections)
		if detections.shape[0] == 0:
			return imgcv
		if self.FLAGS.tracker == "deep_sort":
			scores = np.array(scores)
			features = encoder(imgcv, detections.copy())
			detections = [
			            Detection(bbox, score, feature) for bbox,score, feature in
			            zip(detections,scores, features)]
			# Run non-maxima suppression.
			boxes = np.array([d.tlwh for d in detections])
			scores = np.array([d.confidence for d in detections])
			indices = prep.non_max_suppression(boxes, nms_max_overlap, scores)
			detections = [detections[i] for i in indices]
			tracker.predict()
			tracker.update(detections)
			trackers = tracker.tracks
		elif self.FLAGS.tracker == "sort":
			trackers = tracker.update(detections)
		for track in trackers:

			if self.FLAGS.tracker == "deep_sort":
				if not track.is_confirmed() or track.time_since_update > 1:
					continue
				bbox = track.to_tlbr()
				id_num = str(track.track_id)
				#print(frame_id)
				# if frame_id > 2:
				# 	if result:
				# 		for name in result.keys():
				# 			if name not in names:
				# 				names.append(name)
				# 			if frame_id == result[name][4]:
				# 				loc = result[name]
				# 				x_ = (loc[1]+loc[3])//2
				# 				y_ = (loc[0]+loc[2])//2
				# 				if int(bbox[0]) < x_ and int(bbox[2]) > x_  and int(bbox[1]) < y_ and int(
				# 						bbox[3]) > y_:
				# 					if id_num:
				# 						if a.get(name, '?') == '?':
				# 							a[name] = id_num
				#
				#
				#
				# 						#a.append(b)
				# if names:
				# 	for name_ in names:
				# 		if a.get(name_, '?') != '?':
				# 			if id_num == a[name_]:
				# 				id_num = name_

			#	print(id_num)
				#print(id_num)
				# if id_num == '8':
				# 	id_num = 'azt'
								# print(id_num)

			elif self.FLAGS.tracker == "sort":
				bbox = [int(track[0]), int(track[1]), int(track[2]), int(track[3])]
				id_num = str(int(track[4]))

			if self.FLAGS.csv:
				csv.writerow([frame_id, id_num, int(bbox[0]), int(bbox[1]), int(bbox[2])-int(bbox[0]), int(bbox[3])-int(bbox[1])])
				csv_file.flush()
			if self.FLAGS.display or self.FLAGS.saveVideo:

				cv2.rectangle(imgcv, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), thick//3)
				cv2.putText(imgcv, str(id_num), (int(bbox[0]), int(bbox[1]) - 12), 0, 1e-3 * h, (255, 255, 255), thick//6)


	return imgcv
