import face_recognition
from darkflow.net.build import TFNet
from darkflow.net.yolo import misc
from darkflow.net.yolov2 import predict
import cv2


options = {"model": "/home/ck/darkflow/cfg/stu.cfg", "load": "/home/ck/darkflow/bin/stu_final.weights", "threshold": 0.3}

tfnet = TFNet(options)

#imgcv = cv2.imread("/home/ck/darkflow/sample_img/sample_person.jpg")
tfnet.camera()


# 修改在服务器上，整理过后提交
#tfnet.predict()



#result = tfnet.return_predict(imgcv)

#print(result)
