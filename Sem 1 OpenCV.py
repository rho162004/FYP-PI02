import cv2, time
import numpy as np
import requests

# Quit Program with Proper HouseKeeping
def quit(cap):
    if (cap.isOpened()): cap.release()
    cv2.destroyAllWindows()
    
# Initialise the list of reference points and boolean indicator
mousePt = []
mousePt2 = []
drawPt = []
mouseDown = False
mouseDown2 = False

def onMouseButton(event, x, y, flags, param): # For imgIn
    # Grab references to the global variables
    global mousePt, mouseDown, click, quadrant
    
    # If the left mouse button was clicked, record the (x, y) coordinates and indicate mouse was clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        mousePt = [(x, y)]
        mouseDown = True
        
    # Check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        
    # If the left mouse button was released, record the (x, y) coordinates and indicated mouse was released
        mousePt.append((x, y))
        mouseDown = False
        
        # Only open one quadrant at a time
        if ((x < (width / 2)) & (y < (height / 2)) & 
            (cv2.getWindowProperty(C2, cv2.WND_PROP_VISIBLE) < 1) &
            (cv2.getWindowProperty(C3, cv2.WND_PROP_VISIBLE) < 1) &
            (cv2.getWindowProperty(C4, cv2.WND_PROP_VISIBLE) < 1)): print("Opening " + C1); quadrant = 1
        
        if ((x > (width / 2)) & (y < (height / 2)) &
            (cv2.getWindowProperty(C1, cv2.WND_PROP_VISIBLE) < 1) &
            (cv2.getWindowProperty(C3, cv2.WND_PROP_VISIBLE) < 1) &
            (cv2.getWindowProperty(C4, cv2.WND_PROP_VISIBLE) < 1)): print("Opening " + C2); quadrant = 2
        
        if ((x < (width / 2)) & (y > (height / 2)) &
            (cv2.getWindowProperty(C1, cv2.WND_PROP_VISIBLE) < 1) &
            (cv2.getWindowProperty(C2, cv2.WND_PROP_VISIBLE) < 1) &
            (cv2.getWindowProperty(C4, cv2.WND_PROP_VISIBLE) < 1)): print("Opening " + C3); quadrant = 3
        
        if ((x > (width / 2)) & (y > (height / 2)) &
            (cv2.getWindowProperty(C1, cv2.WND_PROP_VISIBLE) < 1) &
            (cv2.getWindowProperty(C2, cv2.WND_PROP_VISIBLE) < 1) &
            (cv2.getWindowProperty(C3, cv2.WND_PROP_VISIBLE) < 1)): print("Opening " + C4); quadrant = 4
            
        click = True
     
def send_to_telegram(message):
    global msg_sent

    apiToken = '6490993907:AAEL-qoY44V-flvqRHP0TmfItzUBYhLvffI'
    chatID = '4058750208'
    apiURL = f'https://api.telegram.org/bot{apiToken}/sendMessage'
    
    if msg_sent == False:
        try:
            response = requests.post(apiURL, json={'chat_id': chatID, 'text': message})
            print(response.text)
        except Exception as e:
            print(e)
            
    msg_sent = True

     
def detect_objects(img):
    
    blob = cv2.dnn.blobFromImage(img, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layer)#outputs array `
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
                 center_x = int(detection[0] * width)
                 center_y = int(detection[1] * height)
                 w = int(detection[2] * width)
                 h = int(detection[3] * height)
                 # Reactangle Cordinate
                 x = int(center_x - w/2)
                 y = int(center_y - h/2)
                 boxes.append([x, y, w, h])
                 confidences.append(float(confidence))
                 class_ids.append(class_id)
    
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    print('indx:', indexes)
    
    font = cv2.FONT_HERSHEY_SIMPLEX #cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            cx = int((x + x + w) / 2)
            cy = int((y + y + h) / 2)
            label = str(classes[class_ids[i]])
            confidence =  f'{confidences[i]:.2f}' #str(confidences[i])
            # print(label)
            color = (0,255,255) #colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            # cv2.putText(img, label + ' : ' +  confidence , (x + 5, y + 15), font, 0.5, (0,0,0),3)
            cv2.putText(img, label + ' : ' +  confidence , (x + 5, y + 15), font, 0.5, color, 1)
            ClassID.append(class_ids[i])
            CCoords.append([cx, cy])
    for i in CCoords:
        if (i[0] < (width / 2)) & (i[1] < (height / 2)):#q1
            CCoords1.append(i)
            Q1Idx.append(CCoords.index(i))
        if (i[0] > (width / 2)) & (i[1] < (height / 2)):#q2
            CCoords2.append(i)
            Q2Idx.append(CCoords.index(i))
        if (i[0] < (width / 2)) & (i[1] > (height / 2)):                    
            CCoords3.append(i)
            Q3Idx.append(CCoords.index(i))
        if (i[0] > (width / 2)) & (i[1] > (height / 2)):
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
           
def CompwRef(RefClass, LiveClass):#compares ref w live to return missing ids and obstructing ids.       
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
        if 0 <= c < len(classes):
            obs += ', ' + classes[c]
        else:
            print()            
    for c in missing:
        if 0 <= c < len(classes):
            miss += ' ' + classes[c]
        else:
            print()   
    return miss, obs

classesFilename = "./dnn_yolov4/obj.names"
configFilename  = "./dnn_yolov4/yolov4-FYP.cfg"
weightsFilename = "./dnn_yolov4/yolov4-FYP.weights"

net = cv2.dnn.readNetFromDarknet(configFilename, weightsFilename) # this works too
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

classes = []
with open(classesFilename, 'r') as f:
    classes = [line.strip() for line in f.readlines()]
#print("clasees", classes) #prints the names of the classes ina array
layer_name = net.getLayerNames()
output_layer = [layer_name[i - 1] for i in net.getUnconnectedOutLayers()]
                                          
# Initializations, Setups 
cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture(3)
#cap = cv2.VideoCapture("rtsp://192.168.0.105:554/11")
time.sleep(2)

# width = 2560.0/2 
# height = 1440.0/2
width = 1280.0/2
height = 720.0/2
print("setting camera resolution: ", width, height)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print("camera actual resolution received: ", width, height)

width = 1280
height = 720
print("final input image resolution used: ", width, height)

# Initialise Necessary Variables
mainTitle = "Obstruction Detection System"
C1 = "Camera 1"
C2 = "Camera 2"
C3 = "Camera 3"
C4 = "Camera 4"
click = False
msg_sent = False  
scale_percent = 1.5

imgIn = np.zeros((height, width, 3), dtype = "uint8")   # blank images
imgIn2 = np.zeros((height, width, 3), dtype = "uint8")   # blank images
imgInDraw = np.zeros((height, width, 3), dtype = "uint8") # Used for collating all AOI drawings
imgRef = np.zeros((height, width, 3), dtype = "uint8") # Store reference image here
imgMaskDiff = np.zeros((height, width, 3), dtype = "uint8")
imgAverage = np.zeros((height, width, 3), dtype = "float32")

imgTemp = np.ones((int(height/2*scale_percent), int(width/2*scale_percent), 3), dtype = "uint8") # imgTemp used for creating black rectangles on white image
imgTemp2 = np.ones((int(height/2*scale_percent), int(width/2*scale_percent), 3), dtype = "uint8") 
imgTemp3 = np.ones((int(height/2*scale_percent), int(width/2*scale_percent), 3), dtype = "uint8") 
imgTemp4 = np.ones((int(height/2*scale_percent), int(width/2*scale_percent), 3), dtype = "uint8") 

imgMask = np.zeros((int(height/2*scale_percent), int(width/2*scale_percent), 3), dtype = "uint8") # imgMask used for creating green rectangles on black image
imgMask2 = np.zeros((int(height/2*scale_percent), int(width/2*scale_percent), 3), dtype = "uint8") 
imgMask3 = np.zeros((int(height/2*scale_percent), int(width/2*scale_percent), 3), dtype = "uint8") 
imgMask4 = np.zeros((int(height/2*scale_percent), int(width/2*scale_percent), 3), dtype = "uint8") 

quadrant = 0
th = 20
ts = 40
tv = 50
lower = np.array([th, ts, tv]) 
upper = np.array([255, 255, 255])  

# Screen Resolution
x_screen = 1920
y_screen = 1080

# Initialise Classes
class Quadrant:
    def __init__(self, quad_num, x_coord, y_coord, black_rect, green_rect):
        self.quad_num = quad_num
        self.x_coord = x_coord
        self.y_coord = y_coord
        self.black_rect = black_rect
        self.green_rect = green_rect
        
    def onMouseButton_Quadrant(self, event, x, y, flags, param): # Method to draw on resized quadrant
        global mousePt2, mouseDown2, drawPt
        
        if event == cv2.EVENT_LBUTTONDOWN:
            mousePt2 = [(x,y)]
            # Check current quadrant then scale drawn coordinates on imgInDraw
            if quadrant == 1: drawPt = [(int(x/scale_percent), int(y/scale_percent))]
            elif quadrant == 2: drawPt = [(int((x/scale_percent) + (width/2)), int(y/scale_percent))]
            elif quadrant == 3: drawPt = [(int(x/scale_percent), int((y/scale_percent) + (height/2)))]
            elif quadrant == 4: drawPt = [(int((x/scale_percent) + (width / 2)), int((y/scale_percent) + (height/2)))]
            mouseDown2 = True
            
        elif event == cv2.EVENT_LBUTTONUP:
            mousePt2.append((x,y))
            if quadrant == 1: drawPt.append((int(x/scale_percent), int(y/scale_percent)))
            elif quadrant == 2: drawPt.append((int((x/scale_percent) + (width/2)),int(y/scale_percent)))
            elif quadrant == 3: drawPt.append((int(x/scale_percent),int((y/scale_percent) + (height/2))))
            elif quadrant == 4: drawPt.append((int((x/scale_percent) + (width/2)),int((y/scale_percent) + (height/2))))
            mouseDown2 = False
            
            # Drawing on expanded quadrant
            cv2.rectangle(self.black_rect, mousePt2[0], mousePt2[1], (0, 0, 0), 2) # Draw black rectangle
            cv2.rectangle(self.green_rect, mousePt2[0], mousePt2[1], (0, 255, 0), 2) #Draw green rectangle
            
            # Transfer coordinates to ImgInDraw
            cv2.rectangle(imgInDraw, drawPt[0], drawPt[1], (255, 255, 255), -2) # Draw white filled rectangle
        
    def Extract_Quadrant(self): # Method to extract, resize and layer AOI on the extracted quadrant   
                
        # Extracting the quadrant
        crop_width = int(width / 2)
        crop_height = int(height / 2)
        x = self.x_coord
        y = self.y_coord
        crop_Quadrant = imgIn2[y:y+crop_height, x:x+crop_width]
        
        # Resize the quadrant
        width_resize = int(crop_Quadrant.shape[1] * scale_percent)
        height_resize = int(crop_Quadrant.shape[0] * scale_percent)
        dimension = (width_resize, height_resize)
        resized_quadrant = cv2.resize(crop_Quadrant, dimension, interpolation = cv2.INTER_AREA)
        cv2.imshow(self.quad_num, resized_quadrant)
        
        # Check current quadrant and shift resized quadrant on a specific location on screen
        if quadrant == 1: cv2.moveWindow(self.quad_num, 0, 0)
        elif quadrant == 2: cv2.moveWindow(self.quad_num, int(x_screen / 2), 0)
        elif quadrant == 3: cv2.moveWindow(self.quad_num, 0, int((y_screen / 2) - 70))
        elif quadrant == 4: cv2.moveWindow(self.quad_num, int(x_screen / 2), int((y_screen / 2) - 70))
        cv2.setMouseCallback(self.quad_num, self.onMouseButton_Quadrant)
        
        # Layer AOI on the quadrant
        layer = cv2.bitwise_or(self.black_rect, resized_quadrant)
        resized_quadrant = cv2.bitwise_or(layer, self.green_rect) # Convert black rectangle to green rectangle for Area of Interest
        cv2.imshow(self.quad_num, resized_quadrant)
        
cv2.imshow(mainTitle,imgIn)      
cv2.moveWindow(mainTitle, int((x_screen - width) / 2), int((y_screen - height) / 2)) # Set main window at the centre of the screen
cv2.setMouseCallback(mainTitle, onMouseButton)      
  
# Main Loop
bLoop = True
while (bLoop):
    
    # Process Keyboard, Mouse Any Other Inputs
    key = cv2.waitKey(1)
    if (key == 27): bLoop = False ; break
    
    # Keyboard input to reset message sending
    if (key & 0xFF == ord('r')):
        print("Resetting the message alert...")
        msg_sent = False
        print("Resetted.")
    
    # Keyboard inputs for reference image
    if (key & 0xFF == ord('p')): imgRef = imgIn2.copy() # Take Reference Image 
    if (key & 0xFF == ord('l')): cv2.imshow('imgRef', imgRef) # Display Reference Image
    if (key & 0xFF == ord('m')): cv2.destroyWindow("imgRef") # Close Reference Image
    
    # Keyboard input to clear AOI only when in the respective quadrant
    if cv2.getWindowProperty(C1, cv2.WND_PROP_VISIBLE) >= 1:
        if (key & 0xFF == ord('a')): imgMask.fill(0); imgTemp.fill(0); imgInDraw[:int(height/2),:int(width/2)].fill(0)
    if cv2.getWindowProperty(C2, cv2.WND_PROP_VISIBLE) >= 1:
        if (key & 0xFF == ord('s')): imgMask2.fill(0); imgTemp2.fill(0); imgInDraw[:int(height/2),int(width/2):width].fill(0)
    if cv2.getWindowProperty(C3, cv2.WND_PROP_VISIBLE) >= 1:    
        if (key & 0xFF == ord('d')): imgMask3.fill(0); imgTemp3.fill(0); imgInDraw[int(height/2):height,:int(width/2)].fill(0)
    if cv2.getWindowProperty(C4, cv2.WND_PROP_VISIBLE) >= 1:
        if (key & 0xFF == ord('f')): imgMask4.fill(0); imgTemp4.fill(0); imgInDraw[int(height/2):height,int(width/2):width].fill(0)
        
    # Keyboard inputs to close expanded quadrant only when it is opened
    if cv2.getWindowProperty(C1, cv2.WND_PROP_VISIBLE) >= 1:
        if (key & 0xFF == ord('z')): click = False; cv2.destroyWindow(C1)
    if cv2.getWindowProperty(C2, cv2.WND_PROP_VISIBLE) >= 1:
        if (key & 0xFF == ord('x')): click = False; cv2.destroyWindow(C2)
    if cv2.getWindowProperty(C3, cv2.WND_PROP_VISIBLE) >= 1:
        if (key & 0xFF == ord('c')): click = False; cv2.destroyWindow(C3)
    if cv2.getWindowProperty(C4, cv2.WND_PROP_VISIBLE) >= 1:
        if (key & 0xFF == ord('v')): click = False; cv2.destroyWindow(C4)
        
    # Keyboard inputs to only upload and save AOI when on main screen
    if ((cv2.getWindowProperty(C1, cv2.WND_PROP_VISIBLE) < 1) & 
        (cv2.getWindowProperty(C2, cv2.WND_PROP_VISIBLE) < 1) &
        (cv2.getWindowProperty(C3, cv2.WND_PROP_VISIBLE) < 1) &
        (cv2.getWindowProperty(C4, cv2.WND_PROP_VISIBLE) < 1)):
        
        # Upload AOI
        if (key & 0xFF == ord('u')):
        
            print('Uploading Area of Interests...')
        
            # For imgInDraw
            imgInDraw = cv2.imread('imgInDraw.bmp')
            
            # For quadrant 1
            imgTemp = cv2.imread('imgTemp.bmp')
            imgMask = cv2.imread('imgMask.bmp')
            
            # For quadrant 2
            imgTemp2 = cv2.imread('imgTemp2.bmp')
            imgMask2 = cv2.imread('imgMask2.bmp')
            
            # For quadrant 3
            imgTemp3 = cv2.imread('imgTemp3.bmp')
            imgMask3 = cv2.imread('imgMask3.bmp')
            
            # For quadrant 4
            imgTemp4 = cv2.imread('imgTemp4.bmp')
            imgMask4 = cv2.imread('imgMask4.bmp')
            
            print('Area of Interests uploaded.')
            
        # Saving AOI
        if (key & 0xFF == ord('y')):
            
            print('Saving Area of Interests...')
        
            # For imgInDraw
            cv2.imwrite('imgInDraw.bmp', imgInDraw)
            
            # For quadrant 1
            cv2.imwrite('imgTemp.bmp', imgTemp)
            cv2.imwrite('imgMask.bmp', imgMask)
            
            # For quadrant 2
            cv2.imwrite('imgTemp2.bmp', imgTemp2)
            cv2.imwrite('imgMask2.bmp', imgMask2)
            
            # For quadrant 3
            cv2.imwrite('imgTemp3.bmp', imgTemp3)
            cv2.imwrite('imgMask3.bmp', imgMask3)
            
            # For quadrant 4
            cv2.imwrite('imgTemp4.bmp', imgTemp4)
            cv2.imwrite('imgMask4.bmp', imgMask4)
            
            print('Area of Interests saved.')
    
    # # Capture Input Frame
    ret, frame = cap.read()

    if ret: # ie. camera frame is available, do processing here
        
        imgIn = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        imgIn2 = imgIn.copy() # Used for extracting and expanding quadrants
        imgIn3 = imgIn.copy() # Used for taking average
        imgin = imgIn.copy()
        
        if click == True: # Checks when a click on imgIn is detected, extract and expand the quadrant
            if quadrant == 1: Q1 = Quadrant(C1, 0, 0, imgTemp, imgMask); Q1.Extract_Quadrant()
            elif quadrant == 2: Q2 = Quadrant(C2, int(width / 2), 0, imgTemp2, imgMask2); Q2.Extract_Quadrant()
            elif quadrant == 3: Q3 = Quadrant(C3, 0, int(height / 2), imgTemp3, imgMask3); Q3.Extract_Quadrant()
            elif quadrant == 4: Q4 = Quadrant(C4, int(width / 2), int(height / 2), imgTemp4, imgMask4); Q4.Extract_Quadrant()
                
        # Averaging to minimise moving objects as part of difference calculation
        cv2.accumulateWeighted(imgIn3, imgAverage, 0.005) # Taking average of ImgIn3
        imgAverage_uint8 = imgAverage.astype(np.uint8) # Convert averaged ImgIn3 to uint8 data type for bitwise operation
        
        # Extract interested portions of each quadrant
        imgAveragewMask = cv2.bitwise_and(imgAverage_uint8, imgInDraw) # Extracts corresponding pixels bounded within white rectangle
        cv2.accumulateWeighted(imgin, imgAverage, 0.006)
        imgin = imgin.astype(np.uint8)
        imgAvgAoi = cv2.bitwise_and(imgInDraw, imgin)
        imgRefMask = cv2.bitwise_and(imgRef, imgInDraw) # Extracts corresponding pixels bounded within white rectangle
        imgAveragewMaskHSV = cv2.cvtColor(imgAveragewMask, cv2.COLOR_BGR2HSV) # Convert to HSV from BGR
        imgRefMaskHSV = cv2.cvtColor(imgRefMask, cv2.COLOR_BGR2HSV) # Convert to HSV from BGR
        
        H = imgAveragewMaskHSV[:,:,0] # Set H as hue value for imgAveragewMaskHSV
        S = imgAveragewMaskHSV[:,:,1] # Set S as saturation value for imgAveragewMaskHSV
        V = imgAveragewMaskHSV[:,:,2] # Set V as value value for imgAveragewMaskHSV
        
        # Extract H component from imgAveAoI and imgRefwMask
        imgAveragewMaskH = imgAveragewMaskHSV[:,:,0]   
        imgRefMaskH = imgRefMaskHSV[:,:,0]  
        
        dh = cv2.absdiff(imgAveragewMaskH, imgRefMaskH) # Find difference between imgAveragewMaskH & imgRefMaskH
        dh[dh>90] = 180.0 - dh[dh>90]  # hue range is 0-180, so need to correct negative values present in dh, if diff  in hue is greater than 90, correct it i.e. dh = 180 - dh
        imgDiffAbs = cv2.absdiff(imgAveragewMaskHSV, imgRefMaskHSV)
        ds = imgDiffAbs[:,:,1]
        dv = imgDiffAbs[:,:,2]
        
        imgMaskDiff.fill(0)
        
        # test different conditions for creating mask
        imgMaskDiff[dv>tv] = 255
        #imgMask[ds>ts] = 255
        imgMaskDiff[ np.where( (ds>ts) & (V > 20) ) ] = 255
        imgMaskDiff[ np.where( (dh>th) & (S>50) & (V > 50) ) ] = 255
        imgMaskDiffGray = cv2.cvtColor(imgMaskDiff, cv2.COLOR_BGR2GRAY)
        
        RefClassID1, RefClassID2, RefClassID3, RefClassID4 = detect_objects(imgRefMask)
        print('refClass3', RefClassID3)
        LiveClassID1, LiveClassID2, LiveClassID3, LiveClassID4 = detect_objects(imgAvgAoi)#(imgAveragewMask)
        print('LiveClass3', LiveClassID3)
        miss1, obs1 = CompwRef(RefClassID1, LiveClassID1)
        miss2, obs2 = CompwRef(RefClassID2, LiveClassID2)
        miss3, obs3 = CompwRef(RefClassID3, LiveClassID3)
        miss4, obs4 = CompwRef(RefClassID4, LiveClassID4)
        
        contours, hierarchy = cv2.findContours(image=imgMaskDiffGray, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        font = cv2.FONT_HERSHEY_SIMPLEX #cv2.FONT_HERSHEY_PLAIN
        if len(contours) != 0:
            for contour in contours:
                if cv2.contourArea(contour) > 50:
                    x , y, w, h = cv2.boundingRect(contour)
                   # cv2.rectangle(imgIn, (x, y), (x + w, y + h), (0, 0, 255), 3)                 
                    # Finding centre point of image difference
                    M = cv2.moments(contour)
                    cx = int(M["m10"]/M["m00"])
                    cy = int(M["m01"]/M["m00"])
                    # Hightlight quadrant where centre of image difference is detected using its coordinates
                    if (cx < (width / 2)) & (cy < (height / 2)): 
                        if RefClassID1 != LiveClassID1:
                            cv2.rectangle(imgIn, (0, 0), (int(width / 2), int(height / 2)), (0, 0, 255), 5)
                            cv2.rectangle(imgIn, (x, y), (x + w, y + h), (0, 0, 255), 3)
                            cv2.putText(imgIn, miss1 + obs1, (0, 5), font, 0.5, (0, 255, 255), 1) 
                            send_to_telegram("Difference detected!")
                        if RefClassID1 == LiveClassID1:
                            pass
                        if (RefClassID1 == []) & (LiveClassID1==[]):
                            cv2.rectangle(imgIn, (x, y), (x + w, y + h), (0, 0, 255), 3)
                            cv2.rectangle(imgIn, (0, 0), (int(width / 2), int(height / 2)), (0, 0, 255), 5)
                            send_to_telegram("Difference detected!")
                    if (cx > (width / 2)) & (cy < (height / 2)): 
                        if RefClassID2 != LiveClassID2:  
                            cv2.rectangle(imgIn, (int(width / 2), 0), (int(width), int(height / 2)), (0, 0, 255), 5)
                            cv2.putText(imgIn, miss2 + obs2, (645, 5), font, 0.5, (0, 255, 255), 1)
                            cv2.rectangle(imgIn, (0, int(height / 2)), (int(width / 2), int(height)), (0, 0, 255), 5)
                            send_to_telegram("Difference detected!")
                        if RefClassID2 == LiveClassID2:
                            pass
                        if (RefClassID2 == []) & (LiveClassID2==[]):
                            cv2.rectangle(imgIn, (0, int(height / 2)), (int(width / 2), int(height)), (0, 0, 255), 5)
                            cv2.rectangle(imgIn, (x, y), (x + w, y + h), (0, 0, 255), 3)
                            send_to_telegram("Difference detected!")
                    if (cx < (width / 2)) & (cy > (height / 2)): 
                        if RefClassID3 != LiveClassID3:    
                            cv2.rectangle(imgIn, (0, int(height / 2)), (int(width / 2), int(height)), (0, 0, 255), 5)
                            cv2.rectangle(imgIn, (x, y), (x + w, y + h), (0, 0, 255), 3)
                            cv2.putText(imgIn, miss3 + obs3, (0, 365), font, 0.5, (0, 255, 255), 1)    
                            send_to_telegram("Difference detected!")
                        if RefClassID3 == LiveClassID3:
                            pass#print()
                        if (RefClassID3 == []) & (LiveClassID3==[]):
                            cv2.rectangle(imgIn, (0, int(height / 2)), (int(width / 2), int(height)), (0, 0, 255), 5)
                            cv2.rectangle(imgIn, (x, y), (x + w, y + h), (0, 0, 255), 3)
                            send_to_telegram("Difference detected!")
                            
                    if (cx > (width / 2)) & (cy > (height / 2)): 
                        if RefClassID4 != LiveClassID4:
                            cv2.rectangle(imgIn, (int(width / 2), int(height / 2)), (int(width), int(height)), (0, 0, 255), 5)
                            cv2.rectangle(imgIn, (x, y), (x + w, y + h), (0, 0, 255), 3)
                            cv2.putText(imgIn, miss4 + obs4, (645, 365), font, 0.5, (0, 255, 255), 1)
                            send_to_telegram("Difference detected!")
                        if RefClassID4 == LiveClassID4:
                            pass#print()
                        if (RefClassID4 == []) & (LiveClassID4 == []):
                            cv2.rectangle(imgIn, (int(width / 2), int(height / 2)), (int(width), int(height)), (0, 0, 255), 5)
                            cv2.rectangle(imgIn, (x, y), (x + w, y + h), (0, 0, 255), 3)
                            send_to_telegram("Difference detected!")                    
        cv2.imshow(mainTitle, imgIn) 
        cv2.imshow('imgAvgAoi', imgAvgAoi)
        cv2.imshow('imgRefMask', imgRefMask)
# Quit Program with Proper HouseKeeping    
quit(cap)

def imgInGet():
    global imgIn
    return imgIn

def imgAvgAoiGet():
    global imgAvgAoi
    return imgAvgAoi

def imgRefMaskGet():
    global imgRefMask
    return imgRefMask

# Masking Reference - https://www.digitalocean.com/community/tutorials/arithmetic-bitwise-and-masking-python-opencv
# Find Image Difference - https://stackoverflow.com/questions/56183201/detect-and-visualize-differences-between-two-images-with-opencv-python
# Find Contours - https://learnopencv.com/contour-detection-using-opencv-python-c/#:~:text=Find%20the%20Contours,the%20contours%20in%20the%20image.
# Averaging - https://www.geeksforgeeks.org/background-subtraction-in-an-image-using-concept-of-running-average/
# accumulateWeighted - https://docs.opencv.org/2.4/modules/imgproc/doc/motion_analysis_and_object_tracking.html?highlight=accumulate#accumulateweighted
# HSV - https://cvexplained.wordpress.com/2020/04/28/color-detection-hsv/#:~:text=In%20OpenCV%2C%20Hue%20has%20values,255%2C%200%2D255).
# HSV Additional - https://dontrepeatyourself.org/post/color-based-object-detection-with-opencv-and-python/
# Object outline from: https://www.geeksforgeeks.org/find-co-ordinates-of-contours-using-opencv-python/
# Find centre point of contour: https://www.geeksforgeeks.org/python-opencv-find-center-of-contour/
# Find centre point of contours 2: https://learnopencv.com/find-center-of-blob-centroid-using-opencv-cpp-python/
# Package Python Code: https://towardsdatascience.com/how-to-easily-convert-a-python-script-to-an-executable-file-exe-4966e253c7e9
# Open imshow Window at specific coordinates: https://www.geeksforgeeks.org/python-opencv-movewindow-function/
# Check if imshow Window is opened: https://www.geeksforgeeks.org/opencv-python-how-to-detect-if-a-window-is-closed/
# Check if imshow Window is opened 2: https://medium.com/@mh_yip/opencv-detect-whether-a-window-is-closed-or-close-by-press-x-button-ee51616f7088
# Class and Objects: https://www.tutorialspoint.com/python/python_classes_objects.htm#:~:text=Instance%20%E2%88%92%20An%20individual%20object%20of,defined%20in%20a%20class%20definition.
# Telegram Message: https://www.shellhacks.com/python-send-message-to-telegram/#:~:text=Send%20Message%20to%20Telegram%20using%20Python&text=post(apiURL%2C%20json%3D%7B'chat_id,%22Hello%20from%20Python!%22)&text=You%20can%20also%20send%20images,through%20the%20API%20using%20Python
# Find Telegram chat ID: https://www.youtube.com/watch?v=Sq_cYIlm_pM