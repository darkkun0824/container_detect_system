# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\project.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!
import sys
import cv2
import argparse
import random
import numpy as np
import torch
import math

from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import *
from PyQt5 import QtGui


from UI import win  # 界面文件

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
import time

class AppWindow(QMainWindow, win.Ui_MainWindow):
    def __init__(self, parent=None):
        super(AppWindow, self).__init__(parent)
        self.timer_video = QtCore.QTimer()
        self.timer_video_2 = QtCore.QTimer()
        self.cap = cv2.VideoCapture()
        
        self.setupUi(self)
        self.init_slots()
        

        self.model_path = "weights/v5_last.pt"
        self.device = "cuda:0"
        self.conf_thres = 0.5
        self.iou_thres = 0.45  
        
        self.model = attempt_load(self.model_path, map_location=self.device)

        # Get names and colors
        self.stride = int(self.model.stride.max())  # model stride
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
        # print(self.names)

        self.name_pred_dict = {}
        self.name_gt_dict = {}

    def init_slots(self):
        # 默认使用第一个本地camera
        flag = self.cap.open(0)
                
        self.timer_video.timeout.connect(self.show_video_frame)
        self.timer_video.start(30)
        
        self.timer_video_2.timeout.connect(self.check_num)
        self.timer_video_2.start(3000)

    def drawBboxes(self, image, clsId, bbox, conf):
        color = (self.colors[clsId][0], self.colors[clsId][1], self.colors[clsId][2])
        cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)

        label_text = "%s %.1f" % (self.names[clsId], conf * 100)
        label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)

        cv2.rectangle(image, (int(bbox[0]), int(bbox[1])),
                        (int(bbox[0] + label_size[0][0]*1.2), int(bbox[1] + label_size[0][1]*1.4)), color, -1)

        cv2.putText(image, label_text, (int(bbox[0]), int(bbox[1]) + label_size[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255))
        

    def show_video_frame(self):
        flag, img = self.cap.read()
    # img = cv2.imread("/home/qxy/container_detect_system/yolov5-5.0/data/images/微信图片_20240314233439.png")

        if img is not None:
            showimg = img

            start_time = time.time()
            # Padded resize
            img = letterbox(img, 640, stride=self.stride)[0]

            # Convert
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)

            img = torch.from_numpy(img).to(self.device)

            img = img / 255.0  # 0 - 255 to 0.0 - 1.0  

            if len(img.shape) == 3:
                img = img[None]  # expand for batch dim
            
            pred = self.model(img, augment=False)[0]
            
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, None, False)
            
            self.name_pred_dict = {}
            # Process predictions
            for i, det in enumerate(pred):  # per image
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], showimg.shape).round()
                    det = det.cpu().numpy()
                    for *xyxy, conf, cls in reversed(det):
                        if  conf > self.conf_thres:
                            self.drawBboxes(showimg, int(cls), xyxy, conf)
                            
                            if self.names[int(cls)] not in self.name_pred_dict:
                                self.name_pred_dict[self.names[int(cls)]] = 1
                            else:
                                self.name_pred_dict[self.names[int(cls)]] += 1

            dur_time = time.time() - start_time
            fps = int(1 / dur_time)
         
            cv2.putText(showimg, "Camera: 0", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (52, 168, 83),1)
            cv2.putText(showimg, "FPS: {}".format(fps), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (52, 168, 83),1)

            self.result = cv2.cvtColor(showimg, cv2.COLOR_BGR2RGB)
            showImage = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0], self.result.shape[1]*3,
                                     QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(showImage)
            pixmap = pixmap.scaled(self.label_show.size(), Qt.IgnoreAspectRatio)
            self.label_show.setPixmap(pixmap)

        else:
            self.timer_video.stop()
            self.cap.release()
            
            self.label.clear()

    def check_num(self):
        self.name_gt_dict = {}

        if self.lineEdit_7.text().isdigit():
            self.name_gt_dict["vessel"] = int(self.lineEdit_7.text())
        else:
            self.name_gt_dict["vessel"] = 0

        if "vessel" not in self.name_pred_dict:
            self.name_pred_dict["vessel"] = 0

        if self.lineEdit.text().isdigit():
            self.name_gt_dict["label"] = int(self.lineEdit.text())
        else:
            self.name_gt_dict["label"] = 0

        if "label" not in self.name_pred_dict:
            self.name_pred_dict["label"] = 0

        if self.lineEdit_2.text().isdigit():
            self.name_gt_dict["erlenmeyer flask"] = int(self.lineEdit_2.text())
        else:
            self.name_gt_dict["erlenmeyer flask"] = 0

        if "erlenmeyer flask" not in self.name_pred_dict:
            self.name_pred_dict["erlenmeyer flask"] = 0

        if self.lineEdit_3.text().isdigit():
            self.name_gt_dict["drug"] = int(self.lineEdit_3.text())
        else:
            self.name_gt_dict["drug"] = 0

        if "drug" not in self.name_pred_dict:
            self.name_pred_dict["drug"] = 0

        if self.lineEdit_4.text().isdigit():
            self.name_gt_dict["sample vial"] = int(self.lineEdit_4.text())
        else:
            self.name_gt_dict["sample vial"] = 0

        if "sample vial" not in self.name_pred_dict:
            self.name_pred_dict["sample vial"] = 0
                
        if self.lineEdit_5.text().isdigit():
            self.name_gt_dict["beaker"] = int(self.lineEdit_5.text())
        else:
            self.name_gt_dict["beaker"] = 0
            
        if "beaker" not in self.name_pred_dict:
            self.name_pred_dict["beaker"] = 0

        if self.lineEdit_6.text().isdigit():
            self.name_gt_dict["flask"] = int(self.lineEdit_6.text())
        else:
            self.name_gt_dict["flask"] = 0

        if "flask" not in self.name_pred_dict:
            self.name_pred_dict["flask"] = 0

        
        res_list = []
        for pred_name, pred_num in self.name_pred_dict.items():
            if pred_num > self.name_gt_dict[pred_name]:
                res_list.append("more")
            elif pred_num < self.name_gt_dict[pred_name]:
                res_list.append("less")
            else:
                res_list.append("equal")



        if all(x == "equal" for x in res_list):
            self.label_2.setPixmap(QtGui.QPixmap(":/newPrefix/UI/icon/yellow.png"))
            self.label_5.setStyleSheet("QLabel {color: rgb(251, 188, 5);}")
            self.label_3.setPixmap(QtGui.QPixmap(":/newPrefix/UI/icon/gray.png"))
            self.label_6.setStyleSheet("QLabel {color: rgb(0, 0, 0);}")
            self.label.setPixmap(QtGui.QPixmap(":/newPrefix/UI/icon/gray.png"))
            self.label_4.setStyleSheet("QLabel {color: rgb(0, 0, 0);}")

        elif "more" not in res_list and "less" in res_list:
            self.label_3.setPixmap(QtGui.QPixmap(":/newPrefix/UI/icon/green.png"))
            self.label_6.setStyleSheet("QLabel {color: rgb(52, 168, 83);}")
            self.label_2.setPixmap(QtGui.QPixmap(":/newPrefix/UI/icon/gray.png"))
            self.label_5.setStyleSheet("QLabel {color: rgb(0, 0, 0);}")
            self.label.setPixmap(QtGui.QPixmap(":/newPrefix/UI/icon/gray.png"))
            self.label_4.setStyleSheet("QLabel {color: rgb(0, 0, 0);}")

        else:
            self.label.setPixmap(QtGui.QPixmap(":/newPrefix/UI/icon/red.png"))
            self.label_4.setStyleSheet("QLabel {color: rgb(255, 0, 0);}")
            self.label_2.setPixmap(QtGui.QPixmap(":/newPrefix/UI/icon/gray.png"))
            self.label_5.setStyleSheet("QLabel {color: rgb(0, 0, 0);}")
            self.label_3.setPixmap(QtGui.QPixmap(":/newPrefix/UI/icon/gray.png"))
            self.label_6.setStyleSheet("QLabel {color: rgb(0, 0, 0);}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = AppWindow()
    win.show()
    sys.exit(app.exec_())
