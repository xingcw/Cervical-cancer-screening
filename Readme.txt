History operations learning

Test: learning git operations

Undergraduate Thesis Program: Cervical cancer cytologicl AI screening system based on Mask RCNN deep learning model

Abstract:
With the development of computer vision and deep learning, it is possible to use
artificial intelligence to solve the problem of automatic screening of cervical cancer
cells. In the field of automatic cervical cancer cell screening, most research is mainly
based on traditional machine learning algorithms to solve simple cell segmentation and
classification problems, and it is difficult to deal with the problem of cell segmentation
with overlapping normal and abnormal cervical cells.
In this paper, based on the Mask RCNN deep learning model, the cell
segmentation problem in cervical cancer cell screening is studied, overlapping normal
and abnormal cervical cells are segmented by instance segmentation methods, and the
object detection and classification tasks of full-view cell slides are studied.
In terms of cell segmentation, the Mask RCNN model is used to perform instance
segmentation training and prediction on the cervical cell datasets published by ISBI14,
ISBI15, CERVIX93, and the clinical cervical cell datasets produced by the
liquid-based cytology method from hospitals. Finally, the results of the model for
cytoplasm segmentation on the public dataset ISBI14 are an mAP of 0.866, an average
precision of 0.99, and an average recall of 0.993 at the level of IoU = 0.5; results for
nuclei segmentation are an mAP of 0.825, an average precision of 0.99, and an average
recall of 0.998 at the level of IoU = 0.5. The results of the model for cytoplasm
segmentation on the clinical dataset are an mAP of 0.742, an average precision of
0.899, and an average recall of 0.985 at the level of IoU = 0.5.
In terms of object detection, the Mask RCNN model is used to train the dataset
from the 2019 Alibaba Cloud “Digital Body” Visual Challenge, the intelligent
diagnosis of cervical cancer risk. Our results on the binary classification of normal and
abnormal cells are better than those of the top-ranking teams.

Keywords: 
Cervical cancer cells; deep learning; instance segmentation; objects detection

device information: 
pyTorch 1.4.0 + CUDA 10.1
