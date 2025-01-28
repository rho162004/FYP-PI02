import tkinter as tk
import cv2
from PIL import Image, ImageTk
import numpy as np
import time
import datetime
import math
import requests

'''
technical changes to be made:
    allow user to change number of cameras through prompt
    change toggle() function to be more efficient when adding more buttons
    keyboard shortcut all accounted for
'''


def send_to_telegram(text):
    token = "6322081157:AAG_zbPn8b3oXpYBV1IVclsP3IdqlSOxtDI"
    chat_id = "-4058750208"
    url_req = "https://api.telegram.org/bot" + token + \
        "/sendMessage" + "?chat_id=" + chat_id + "&text=" + text
    results = requests.get(url_req)
    print(results.json())


# capture and process video through here
class MyVideoCapture:
    def __init__(self, quad_num, video_source=1):
        global num_cameras
        self.cams = num_cameras

        # load YOLOv4
        classesFilename = "./dnn_yolov4/obj.names"
        configFilename = "./dnn_yolov4/yolov4-FYP.cfg"
        weightsFilename = "./dnn_yolov4/yolov4-FYP.weights"

        # start up video
        self.vid = cv2.VideoCapture(video_source)
        ret, frame = self.vid.read()
        self.net = cv2.dnn.readNetFromDarknet(configFilename, weightsFilename)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.classes = self.load_classes(classesFilename)

        # set variables
        self.width = 1280.0/2
        self.height = 720.0/2
        width = self.width
        height = self.height

        self.quad_num = quad_num

        self.mainTitle = "Obstruction Destruction System"
        self.C1 = "Camera 1"
        self.C2 = "Camera 2"
        self.C3 = "Camera 3"
        self.C4 = "Camera 4"

        # fetch ref and mask from saved images
        self.scale_percent = 1.0
        self.ref_title = "refs/ref"+str(self.quad_num)+".bmp"
        self.mask_title = "refs/imgMask"+str(self.quad_num)+".bmp"
        self.temp_title = "refs/imgTemp"+str(self.quad_num)+".bmp"
        self.indraw_title = "refs/imgInDraw"+str(self.quad_num)+".bmp"

        # image declaration :heart_eyes:
        self.imgIn = np.zeros((int(height), int(width), 3),
                              dtype="uint8")   # blank images
        self.imgIn2 = np.zeros((int(height), int(width), 3),
                               dtype="uint8")   # blank images

        # Used for collating all AOI drawings
        self.imgInDraw = cv2.imread(self.indraw_title)
        # Store reference image here
        self.imgRef = cv2.imread(self.ref_title)
        self.imgMaskDiff = np.zeros(
            (int(height), int(width), 3), dtype="uint8")   # blank images
        self.imgAverage = np.zeros(
            (int(height), int(width), 3), dtype="float32")   # blank images

        # imgTemp used for creating black rectangles on white image
        self.imgTemp = cv2.imread(self.temp_title)

        # imgMask used for creating green rectangles on black image
        self.imgMask = cv2.imread(self.mask_title)

        # (relatively unused) variable declaration
        self.th = 20
        self.ts = 40
        self.tv = 50
        self.lower = np.array([self.th, self.ts, self.tv])
        self.upper = np.array([255, 255, 255])

        # screen res
        width = 1280.0/2
        height = 720.0/2

        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

        # drawing rectangles
        self.ix = 0
        self.endx = 0
        self.iy = 0
        self.endy = 0
        
        self.timer = datetime.datetime(2027, 1, 1, 1, 1, 1, 1)
        self.current = datetime.datetime.now()
        self.timesup = True
        self.seconds = 10

    # run upon destruction of object
    def __del__(self):
        # Release the video source when the object is destroyed
        if self.vid.isOpened():
            self.vid.release()
            cv2.destroyAllWindows()
            print("Stream ended")

    # load yolo classes
    def load_classes(self, classes_filename):
        self.classes = []
        with open(classes_filename, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.layer_name = self.net.getLayerNames()
        self.output_layer = [self.layer_name[i - 1]
                             for i in self.net.getUnconnectedOutLayers()]
        return self.classes

    # capture reference image
    def ref_capture(self):
        # same name as a function in App that calls this function
        self.imgRef = self.imgIn2.copy()
        print("Captured reference frame")
        cv2.imwrite(self.ref_title, self.imgRef)

    # clear reference mask
    def clearmask(self):
        self.imgMask.fill(0)
        self.imgTemp.fill(0)
        self.imgInDraw.fill(0)
        print("Mask cleared")

        cv2.imwrite(self.mask_title, self.imgMask)
        cv2.imwrite(self.temp_title, self.imgTemp)
        cv2.imwrite(self.indraw_title, self.imgInDraw)

    # split the input frame into individual cameras
    def split_frame(self, merged_frame):
        height, width, _ = merged_frame.shape
        #print("shape h:", height, "shape w:", width)

        # assuming that the number of squares is a square number (height and width of grid is same number of images)
        sqsize = int(math.sqrt(self.cams))  # half height
        sqsize = int(math.sqrt(self.cams))  # half width

        col = int(self.quad_num % sqsize)
        row = int(self.quad_num / sqsize)

        hheight = height/sqsize
        hwidth = width/sqsize

        r1 = int((hheight*row))
        r2 = int(hheight*(row+1))
        c1 = int((hwidth*col))
        c2 = int(hwidth*(col+1))

        quad = merged_frame[r1:r2, c1:c2]
        return quad

    # YOLO things
    def detect_objects(self, img):
        blob = cv2.dnn.blobFromImage(
            img, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layer)  # outputs array `
        # print('outs', outs)

        # Showing Information on the screen
        class_ids = []
        confidences = []
        boxes = []

        CCoords = []
        ClassID = []

        ClassID1 = []
        ClassID2 = []
        ClassID3 = []
        ClassID4 = []

        CCoords1 = []
        CCoords2 = []
        CCoords3 = []
        CCoords4 = []

        Q1Idx = []
        Q2Idx = []
        Q3Idx = []
        Q4Idx = []

        for out in outs:
            for detection in out:

                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detection
                    center_x = int(detection[0] * self.width)
                    center_y = int(detection[1] * self.height)
                    w = int(detection[2] * self.width)
                    h = int(detection[3] * self.height)
                    # Reactangle Cordinate
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        print('indx:', indexes)

        font = cv2.FONT_HERSHEY_SIMPLEX  # cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                cx = int((x + x + w) / 2)
                cy = int((y + y + h) / 2)
                label = str(self.classes[class_ids[i]])
                confidence = f'{confidences[i]:.2f}'  # str(confidences[i])
                # print(label)
                color = (0, 255, 255)  # colors[i]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                # cv2.putText(img, label + ' : ' +  confidence , (x + 5, y + 15), font, 0.5, (0,0,0),3)
                cv2.putText(img, label + ' : ' + confidence,
                            (x + 5, y + 15), font, 0.5, color, 1)
                ClassID.append(class_ids[i])
                CCoords.append([cx, cy])
        for i in CCoords:
            if (i[0] < (self.width)) & (i[1] < (self.height)):  # q1
                CCoords1.append(i)
                Q1Idx.append(CCoords.index(i))
            if (i[0] > (self.width / 2)) & (i[1] < (self.height / 2)):  # q2
                CCoords2.append(i)
                Q2Idx.append(CCoords.index(i))
            if (i[0] < (self.width / 2)) & (i[1] > (self.height / 2)):  # q3
                CCoords3.append(i)
                Q3Idx.append(CCoords.index(i))
            if (i[0] > (self.width / 2)) & (i[1] > (self.height / 2)):  # q4
                CCoords4.append(i)
                Q4Idx.append(CCoords.index(i))
        for idx in Q1Idx:
            if 0 <= idx < len(ClassID):
                ClassID1.append(ClassID[idx])
            else:
                print()
        for idx in Q2Idx:
            if 0 <= idx < len(ClassID):
                ClassID2.append(ClassID[idx])
            else:
                print()
        for idx in Q3Idx:
            if 0 <= idx < len(ClassID):
                ClassID3.append(ClassID[idx])
            else:
                print()
        for idx in Q4Idx:
            if 0 <= idx < len(ClassID):
                ClassID4.append(ClassID[idx])
            else:
                print()
        return ClassID1, ClassID2, ClassID3, ClassID4

    # compares ref w live to return missing ids and obstructing ids.
    def CompwRef(self, RefClass, LiveClass):
        miss = ' missing: '
        obs = ' obstruction: '
        obstruction = []
        missing = []
        for item in RefClass[:]:
            if item in LiveClass:
                LiveClass.remove(item)
                RefClass.remove(item)
                obstruction = LiveClass.copy()
                missing = RefClass.copy()
            elif item not in LiveClass:
                missing = RefClass.copy()
                obstruction = LiveClass.copy()
        obstruction = LiveClass.copy()
        missing = RefClass.copy()
        for c in obstruction:
            if 0 <= c < len(self.classes):
                obs += ', ' + self.classes[c]
            else:
                print()
        for c in missing:
            if 0 <= c < len(self.classes):
                miss += ' ' + self.classes[c]
            else:
                print()
        return miss, obs

    def get_frame(self):
        self.current = datetime.datetime.now()
        if self.vid.isOpened():
            width = 1280.0/2
            height = 720.0/2
            ret, frame = self.vid.read()
            if ret:
                iwidth = int(width)
                iheight = int(height)
                frame = self.split_frame(frame)
                self.imgIn = cv2.resize(
                    frame, (iwidth, iheight), interpolation=cv2.INTER_AREA)
                self.imgIn2 = self.imgIn.copy()  # Used for extracting and expanding quadrants
                self.imgIn3 = self.imgIn.copy()  # Used for taking average
                self.imgin = self.imgIn.copy()

                # Averaging to minimise moving objects as part of difference calculation
                # Taking average of ImgIn3
                cv2.accumulateWeighted(self.imgIn3, self.imgAverage, 0.005)
                # Convert averaged ImgIn3 to uint8 data type for bitwise operation
                imgAverage_uint8 = self.imgAverage.astype(np.uint8)

                while(self.ix > width):
                    self.ix = self.ix-width
                while(self.endx > width):
                    self.endx = self.endx-width
                while(self.iy > height):
                    self.iy = self.iy-height
                while(self.endy > height):
                    self.endy = self.endy-height

                if (self.ix > 0) and (self.endx > 0):
                    self.imgInDraw = cv2.rectangle(
                        self.imgInDraw, (int(self.ix), int(self.iy)), (int(self.endx), int(self.endy)), (255, 0, 255), -1)
                    cv2.imwrite(self.indraw_title, self.imgInDraw)

                # Extract interested portions of each quadrant
                # Extracts corresponding pixels bounded within white rectangle
                imgAveragewMask = cv2.bitwise_and(
                    imgAverage_uint8, self.imgInDraw)
                cv2.accumulateWeighted(self.imgin, self.imgAverage, 0.006)
                self.imgin = self.imgin.astype(np.uint8)
                imgAvgAoi = cv2.bitwise_and(self.imgInDraw, self.imgin)
                # Extracts corresponding pixels bounded within white rectangle
                imgRefMask = cv2.bitwise_and(self.imgRef, self.imgInDraw)
                imgAveragewMaskHSV = cv2.cvtColor(
                    imgAveragewMask, cv2.COLOR_BGR2HSV)  # Convert to HSV from BGR
                imgRefMaskHSV = cv2.cvtColor(
                    imgRefMask, cv2.COLOR_BGR2HSV)  # Convert to HSV from BGR

                # Set H as hue value for imgAveragewMaskHSV
                H = imgAveragewMaskHSV[:, :, 0]
                # Set S as saturation value for imgAveragewMaskHSV
                S = imgAveragewMaskHSV[:, :, 1]
                # Set V as value value for imgAveragewMaskHSV
                V = imgAveragewMaskHSV[:, :, 2]

                # Extract H component from imgAveAoI and imgRefwMask
                imgAveragewMaskH = imgAveragewMaskHSV[:, :, 0]
                imgRefMaskH = imgRefMaskHSV[:, :, 0]

                # Find difference between imgAveragewMaskH & imgRefMaskH
                dh = cv2.absdiff(imgAveragewMaskH, imgRefMaskH)
                # hue range is 0-180, so need to correct negative values present in dh, if diff  in hue is greater than 90, correct it i.e. dh = 180 - dh
                dh[dh > 90] = 180.0 - dh[dh > 90]
                imgDiffAbs = cv2.absdiff(imgAveragewMaskHSV, imgRefMaskHSV)
                ds = imgDiffAbs[:, :, 1]
                dv = imgDiffAbs[:, :, 2]

                self.imgMaskDiff.fill(0)

                # test different conditions for creating mask
                self.imgMaskDiff[dv > self.tv] = 255
                #imgMask[ds>ts] = 255
                self.imgMaskDiff[np.where((ds > self.ts) & (V > 20))] = 255
                self.imgMaskDiff[np.where(
                    (dh > self.th) & (S > 50) & (V > 50))] = 255
                imgMaskDiffGray = cv2.cvtColor(
                    self.imgMaskDiff, cv2.COLOR_BGR2GRAY)

                RefClassID1, RefClassID2, RefClassID3, RefClassID4 = self.detect_objects(
                    imgRefMask)
                #print('refClass3', RefClassID3)
                LiveClassID1, LiveClassID2, LiveClassID3, LiveClassID4 = self.detect_objects(
                    imgAvgAoi)  # (imgAveragewMask)
                #print('LiveClass3', LiveClassID3)
                miss1, obs1 = self.CompwRef(RefClassID1, LiveClassID1)
                miss2, obs2 = self.CompwRef(RefClassID2, LiveClassID2)
                miss3, obs3 = self.CompwRef(RefClassID3, LiveClassID3)
                miss4, obs4 = self.CompwRef(RefClassID4, LiveClassID4)

                contours, hierarchy = cv2.findContours(
                    image=imgMaskDiffGray, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
                font = cv2.FONT_HERSHEY_SIMPLEX  # cv2.FONT_HERSHEY_PLAIN
                if len(contours) != 0:
                    for contour in contours:
                        if cv2.contourArea(contour) > 50:
                            x, y, w, h = cv2.boundingRect(contour)
                           # cv2.rectangle(imgIn, (x, y), (x + w, y + h), (0, 0, 255), 3)
                            # Finding centre point of image difference
                            M = cv2.moments(contour)
                            cx = int(M["m10"]/M["m00"])
                            cy = int(M["m01"]/M["m00"])

                            width = 1280
                            height = 720

                            msg = "Difference detected in quadrant" + str(self.quad_num+1) + "!"

                            # Hightlight quadrant where centre of image difference is detected using its coordinates
                            if (cx < (width)) & (cy < (height)):
                                if RefClassID1 != LiveClassID1:
                                    cv2.rectangle(
                                        self.imgIn, (0, 0), (int(width / 2), int(height / 2)), (0, 0, 255), 5)
                                    cv2.rectangle(
                                        self.imgIn, (x, y), (x + w, y + h), (0, 0, 255), 3)
                                    # cv2.putText(
                                    #    self.imgIn, miss1 + obs1, (0, 5), font, 0.5, (0, 255, 255), 1)
                                    if self.timesup:
                                        send_to_telegram(msg)
                                        self.timer = self.current + datetime.timedelta(seconds=self.seconds)
                                        self.timesup = False
                                        print("msg send, timer started")
                                    else:
                                        if (self.current>= self.timer):
                                            self.timesup = True
                                if RefClassID1 == LiveClassID1:
                                    pass
                                if (RefClassID1 == []) & (LiveClassID1 == []):
                                    # self.countdown1()
                                    # if self.countedQ1:
                                    cv2.rectangle(
                                        self.imgIn, (x, y), (x + w, y + h), (0, 0, 255), 3)
                                    cv2.rectangle(
                                        self.imgIn, (0, 0), (int(width / 2), int(height / 2)), (0, 0, 255), 5)
                                    #send_to_telegram("Difference detected!")
                                    # else:
                                    # self.countedQ1=False
                            if (cx > (width / 2)) & (cy < (height / 2)):
                                if RefClassID2 != LiveClassID2:
                                    cv2.rectangle(
                                        self.imgIn, (int(width / 2), 0), (int(width), int(height / 2)), (0, 0, 255), 5)
                                    # cv2.putText(
                                    #    self.imgIn, miss2 + obs2, (645, 5), font, 0.5, (0, 255, 255), 1)
                                    cv2.rectangle(
                                        self.imgIn, (0, int(height / 2)), (int(width / 2), int(height)), (0, 0, 255), 5)
                                    #msg = msg + " liveclass2"
                                    #send_to_telegram(msg)
                                if RefClassID2 == LiveClassID2:
                                    pass
                                if (RefClassID2 == []) & (LiveClassID2 == []):
                                    cv2.rectangle(
                                        self.imgIn, (0, int(height / 2)), (int(width / 2), int(height)), (0, 0, 255), 5)
                                    cv2.rectangle(
                                        self.imgIn, (x, y), (x + w, y + h), (0, 0, 255), 3)
                                    #send_to_telegram("Difference detected!")
                            if (cx < (width / 2)) & (cy > (height / 2)):
                                if RefClassID3 != LiveClassID3:
                                    cv2.rectangle(
                                        self.imgIn, (0, int(height / 2)), (int(width / 2), int(height)), (0, 0, 255), 5)
                                    cv2.rectangle(
                                        self.imgIn, (x, y), (x + w, y + h), (0, 0, 255), 3)
                                    # cv2.putText(
                                    #     self.imgIn, miss3 + obs3, (0, 365), font, 0.5, (0, 255, 255), 1)
                                    #msg = msg + " liveclass3"
                                    #send_to_telegram(msg)
                                if RefClassID3 == LiveClassID3:
                                    pass  # print()
                                if (RefClassID3 == []) & (LiveClassID3 == []):
                                    cv2.rectangle(
                                        self.imgIn, (0, int(height / 2)), (int(width / 2), int(height)), (0, 0, 255), 5)
                                    cv2.rectangle(
                                        self.imgIn, (x, y), (x + w, y + h), (0, 0, 255), 3)
                                    #send_to_telegram("Difference detected!")

                            if (cx > (width / 2)) & (cy > (height / 2)):
                                if RefClassID4 != LiveClassID4:
                                    cv2.rectangle(self.imgIn, (int(
                                        width / 2), int(height / 2)), (int(width), int(height)), (0, 0, 255), 5)
                                    cv2.rectangle(
                                        self.imgIn, (x, y), (x + w, y + h), (0, 0, 255), 3)
                                    # cv2.putText(
                                    #    self.imgIn, miss4 + obs4, (645, 365), font, 0.5, (0, 255, 255), 1)
                                    #msg = msg + " liveclass4"
                                    #send_to_telegram(msg)
                                if RefClassID4 == LiveClassID4:
                                    pass  # print()
                                if (RefClassID4 == []) & (LiveClassID4 == []):
                                    cv2.rectangle(self.imgIn, (int(
                                        width / 2), int(height / 2)), (int(width), int(height)), (0, 0, 255), 5)
                                    cv2.rectangle(
                                        self.imgIn, (x, y), (x + w, y + h), (0, 0, 255), 3)
                                    #send_to_telegram("Difference detected!")

                return (ret, cv2.cvtColor(self.imgIn, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (None, None)


class App:
    def __init__(self, window):
        self.window = window
        self.window.title("FYP PI02")

        self.active_quad = 0  # set active quadrant

        self.vid = cv2.VideoCapture(1)

        self.video_captures = []

        self.video_captures.append(MyVideoCapture(0))
        self.video_captures.append(MyVideoCapture(1))
        self.video_captures.append(MyVideoCapture(2))
        self.video_captures.append(MyVideoCapture(3))

        # Use the fixed window size
        self.canvas = tk.Canvas(window, width=1280, height=720)
        self.canvas.pack()

        self.canvas.bind('<Motion>', self.mouseMove)
        self.canvas.bind("<Button-1>", self.startpaint)
        self.canvas.bind("<ButtonRelease-1>", self.endpaint)

        # buttons
        self.btn_clearmask = tk.Button(
            window, text="Clear Mask", width=30, command=self.clearmask)
        self.btn_clearmask.pack(side=tk.RIGHT, anchor=tk.NE, expand=True)
        self.btn_snapshot = tk.Button(
            window, text="Take Reference", width=30, command=self.ref_capture)
        self.btn_snapshot.pack(side=tk.RIGHT, anchor=tk.NE, expand=True)
        self.btn_viewref = tk.Button(
            window, text="View Reference", width=20, command=self.show_ref)
        self.btn_viewref.pack(side=tk.RIGHT, anchor=tk.NE, expand=True)

        # quad buttons
        self.btn_Q1 = tk.Button(
            window, text="Q1", width=15, relief="raised", command=lambda: self.toggle(1))
        self.btn_Q2 = tk.Button(
            window, text="Q2", width=15, relief="raised", command=lambda: self.toggle(2))
        self.btn_Q3 = tk.Button(
            window, text="Q3", width=15, relief="raised", command=lambda: self.toggle(3))
        self.btn_Q4 = tk.Button(
            window, text="Q4", width=15, relief="raised", command=lambda: self.toggle(4))

        time.sleep(2)

        self.btn_Q1.pack(anchor=tk.CENTER, expand=True)
        self.btn_Q2.pack(anchor=tk.CENTER, expand=True)
        self.btn_Q3.pack(anchor=tk.CENTER, expand=True)
        self.btn_Q4.pack(anchor=tk.CENTER, expand=True)

        # after called once, update auto called
        self.delay = 15
        self.update()

        self.window.mainloop()

    def show_ref(self):
        title = "Reference " + str(self.active_quad)
        cv2.imshow(title, self.video_captures[self.active_quad].imgRef)

    def ref_capture(self):
        # same name as a function in MyVideoCapture
        # needs to call seperately and not from buttons bc it doesn't recheck self.active_quad
        self.video_captures[self.active_quad].ref_capture()

    def toggle(self, quad):
        if quad == 1:
            self.btn_Q1.config(relief="sunken")
            self.btn_Q2.config(relief="raised")
            self.btn_Q3.config(relief="raised")
            self.btn_Q4.config(relief="raised")
        elif quad == 2:
            self.btn_Q1.config(relief="raised")
            self.btn_Q2.config(relief="sunken")
            self.btn_Q3.config(relief="raised")
            self.btn_Q4.config(relief="raised")
        elif quad == 3:
            self.btn_Q1.config(relief="raised")
            self.btn_Q2.config(relief="raised")
            self.btn_Q3.config(relief="sunken")
            self.btn_Q4.config(relief="raised")
        elif quad == 4:
            self.btn_Q1.config(relief="raised")
            self.btn_Q2.config(relief="raised")
            self.btn_Q3.config(relief="raised")
            self.btn_Q4.config(relief="sunken")

        # 0 indexed array of video_captures vs 1 indexed quad in this function
        self.active_quad = quad-1

    def clearmask(self):
        self.video_captures[self.active_quad].clearmask()

    def mouseMove(self, e):
        x = e.x
        y = e.y
        #print("Mouse: ", x, y)

    def startpaint(self, event):
        self.video_captures[self.active_quad].iy
        self.video_captures[self.active_quad].iy, self.video_captures[self.active_quad].ix, self.video_captures[
            self.active_quad].endx, self.video_captures[self.active_quad].endy

        print("Start rectangle at:", self.video_captures[self.active_quad].ix,
              self.video_captures[self.active_quad].iy, self.video_captures[self.active_quad].endx, self.video_captures[self.active_quad].endy)
        self.drawing = True
        self.video_captures[self.active_quad].ix, self.video_captures[self.active_quad].iy = (
            event.x, event.y)

    def endpaint(self, event):
        self.video_captures[self.active_quad].endx, self.video_captures[self.active_quad].endy

        print("End rectangle at:",
              self.video_captures[self.active_quad].endx, self.video_captures[self.active_quad].endy)
        self.drawing = False
        self.video_captures[self.active_quad].endx, self.video_captures[self.active_quad].endy = (
            event.x, event.y)

    def update(self):
        image_frames = []

        for video in self.video_captures:
            ret, frame = video.get_frame()
            if ret:
                image_frames.append(frame)

        merged_image = self.assemble_grid(image_frames)
        merged_image = Image.fromarray(merged_image)

        self.photo = ImageTk.PhotoImage(image=merged_image)
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(self.delay, self.update)

    def concat_vh(self, list_2d):
        # return final image
        return cv2.vconcat([cv2.hconcat(list_h)
                            for list_h in list_2d])

    def assemble_grid(self, images):
        num_images = len(images)
        grid_size = int(np.sqrt(num_images))

        # Manually construct grid without NumPy reshape:
        merged_image = None
        for row_index in range(grid_size):
            row_images = images[row_index *
                                grid_size: (row_index + 1) * grid_size]
            concatenated_row = cv2.hconcat(row_images)
            if merged_image is None:
                merged_image = concatenated_row
            else:
                merged_image = cv2.vconcat([merged_image, concatenated_row])

        return merged_image


def main():
    global num_cameras
    num_cameras = 4
    root = tk.Tk()
    app = App(root)
    root.mainloop()


if __name__ == "__main__":
    main()

