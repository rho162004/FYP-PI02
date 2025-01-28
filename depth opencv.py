import numpy as np
import cv2 
import glob
import math
from imutils import paths 
import imutils


#camera calibrate and depth

#steps: get right/left cam images of checkerboard 
#       calibrate each cam indiv
#       stereocalibrate
#       stereorectify
#       depth map

#simplified MYVIDEOCAPTURE for fetching individual cameras
class MyVideoCapture:
    def __init__(self, quad_num, video_source=1):
        self.vid = cv2.VideoCapture(video_source)
        ret, frame = self.vid.read()

        self.width = 1280.0/2
        self.height = 720.0/2
        width = self.width
        height = self.height
        self.quad_num = quad_num
        self.mainTitle = "Depth Test"

        # image declaration :heart_eyes:
        self.imgIn = np.zeros((int(height), int(width), 3),
                              dtype="uint8")   # blank images
        self.imgIn2 = np.zeros((int(height), int(width), 3),
                               dtype="uint8")   # blank images
        
        # screen res
        width = 1280.0/2
        height = 720.0/2

        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

        # screen res
        self.width = 1280.0/2
        self.height = 720.0/2

    # run upon destruction of object
    def __del__(self):
        # Release the video source when the object is destroyed
        if self.vid.isOpened():
            self.vid.release()
            cv2.destroyAllWindows()
            print("Stream ended")
            
    def split_frame(self, merged_frame):
        height, width, _ = merged_frame.shape
        #print("shape h:", height, "shape w:", width)

        # assuming that the number of squares is a square number (height and width of grid is same number of images)
        sqsize = int(math.sqrt(4))  # half height
        sqsize = int(math.sqrt(4))  # half width

        col = int(self.quad_num % sqsize)
        row = int(self.quad_num / sqsize)
        
        hheight = height/sqsize
        hwidth = width/sqsize
        
        r1 = int((hheight*row))
        r2 = int(hheight*(row+1))
        c1 = int((hwidth*col))
        c2 = int(hwidth*(col+1))
        
        print(hheight, hwidth)
        print(r1, r2, c1, c2, "quad", self.quad_num, "col:", col, "row:", row)
 
        quad = merged_frame[r1:r2, c1:c2]
        return quad

    def get_frame(self):
        if self.vid.isOpened():
            width = self.width
            height = self.height
            ret, frame = self.vid.read()
            if ret:
                iwidth = int(width)
                iheight = int(height)
                frame = self.split_frame(frame)
                self.imgIn = cv2.resize(frame, (iwidth, iheight), interpolation=cv2.INTER_AREA)
                cropHorizontal(self.imgIn)
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (None, None)

CROP_WIDTH = 960
CAMERA_WIDTH=1280
CAMERA_HEIGHT=720


def cropHorizontal(image):
    return image[:,
                 int((CAMERA_WIDTH-CROP_WIDTH)/2):
                 int(CROP_WIDTH+(CAMERA_WIDTH-CROP_WIDTH)/2)]
        
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('cropped/*.png')

for fname in images:   
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7,6), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (7,6), corners2, ret)
        cv2.imshow("fname", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows
    else:
        print(ret)
        cv2.imshow("fname", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows

#calibrate camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
img = cv2.imread('somework/1.png')
h,  w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

'''
#remap to remove distoriton
mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]

#mean error
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "total error: {}".format(mean_error/len(objpoints)) )
'''


q1 = MyVideoCapture(0)
q2 = MyVideoCapture(1)

ret, frame = q1.get_frame()
h,w,_ = frame.shape

#calibrate camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

#remap to remove distoriton
mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

cv2.imshow("left", frame)
cv2.waitKey(0)
cv2.destroyAllWindows

h, w, _ = frame.shape
tx, ty = 20, 0
translation_matrix = np.array([
    [1, 0, tx],
    [0, 1, ty]
], dtype = np.float32)
imgRight = cv2.warpAffine(src=frame, M=translation_matrix, dsize=(w, h))

tx, ty = -20, 0
translation_matrix = np.array([
    [1, 0, tx],
    [0, 1, ty]
], dtype = np.float32)
imgLeft = cv2.warpAffine(src=frame, M=translation_matrix, dsize=(w,h))

cv2.imshow("left", imgLeft)
cv2.imshow("right", imgRight)
cv2.waitKey(0)
cv2.destroyAllWindows


imgLeft=cv2.resize(cv2.cvtColor(imgLeft, cv2.COLOR_BGR2GRAY), (640, 360), cv2.INTER_AREA)
imgRight=cv2.resize(cv2.cvtColor(imgRight, cv2.COLOR_BGR2GRAY), (640, 360), cv2.INTER_AREA)

def ShowDisparity(bSize=19):
    stereo = cv2.StereoBM_create(numDisparities=32, blockSize=bSize)
    
    disparity = stereo.compute(imgLeft, imgRight)
    
    min = disparity.min()
    max = disparity.max()
    disparity = np.uint8(255*(disparity-min)/(max-min))
    
    return disparity


result = ShowDisparity()
cv2.imshow("test1", result)
cv2.imshow("test2", imgRight)
cv2.waitKey(0)
cv2.destroyAllWindows()


#disparity ref: https://medium.com/analytics-vidhya/distance-estimation-cf2f2fd709d8
#calibration ref: https://albertarmea.com/post/opencv-stereo-camera/
#horizontal translation ref: https://learnopencv.com/image-rotation-and-translation-using-opencv/
#note: ref images and camera are cropped due to too extreme fisheye effect on the edge
