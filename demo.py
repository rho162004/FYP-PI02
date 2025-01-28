import tkinter as tk
import cv2
from PIL import Image, ImageTk
import numpy as np
import time
import math

class MyVideoCapture:
    def __init__(self, quad_num, size, video_source=1):
        self.vid = cv2.VideoCapture(video_source)
        ret, frame = self.vid.read()

        self.width = 1280.0/2
        self.height = 720.0/2
        width = self.width
        height = self.height

        self.quad_num = quad_num
        self.size = size

        self.mainTitle = "Obstruction Destruction System"
        self.C1 = "Camera 1"
        self.C2 = "Camera 2"
        self.C3 = "Camera 3"
        self.C4 = "Camera 4"

        # not declaring click, msg_sent because not needed rn
        self.scale_percent = 1.0

        # image declaration :heart_eyes:
        self.imgIn = np.zeros((int(height), int(width), 3),
                              dtype="uint8")   # blank images
        self.imgIn2 = np.zeros((int(height), int(width), 3),
                               dtype="uint8")   # blank images
        # Used for collating all AOI drawings
        self.imgInDraw = np.zeros(
            (int(height), int(width), 3), dtype="uint8")   # blank images
        # Store reference image here
        self.imgRef = np.zeros((int(height), int(width), 3),
                               dtype="uint8")   # blank images
        self.imgMaskDiff = np.zeros(
            (int(height), int(width), 3), dtype="uint8")   # blank images
        self.imgAverage = np.zeros(
            (int(height), int(width), 3), dtype="float32")   # blank images

        # imgTemp used for creating black rectangles on white image
        self.imgTemp = np.ones(
            (int(height/2*self.scale_percent), int(width/2*self.scale_percent), 3), dtype="uint8")
        self.imgTemp2 = np.ones(
            (int(height/2*self.scale_percent), int(width/2*self.scale_percent), 3), dtype="uint8")
        self.imgTemp3 = np.ones(
            (int(height/2*self.scale_percent), int(width/2*self.scale_percent), 3), dtype="uint8")
        self.imgTemp4 = np.ones(
            (int(height/2*self.scale_percent), int(width/2*self.scale_percent), 3), dtype="uint8")

        # imgMask used for creating green rectangles on black image
        self.imgMask = np.zeros(
            (int(height/2*self.scale_percent), int(width/2*self.scale_percent), 3), dtype="uint8")
        self.imgMask2 = np.zeros(
            (int(height/2*self.scale_percent), int(width/2*self.scale_percent), 3), dtype="uint8")
        self.imgMask3 = np.zeros(
            (int(height/2*self.scale_percent), int(width/2*self.scale_percent), 3), dtype="uint8")
        self.imgMask4 = np.zeros(
            (int(height/2*self.scale_percent), int(width/2*self.scale_percent), 3), dtype="uint8")

        self.th = 20
        self.ts = 40
        self.tv = 50
        self.lower = np.array([self.th, self.ts, self.tv])
        self.upper = np.array([255, 255, 255])

        # screen res
        self.x_screen = 1280
        self.y_screen = 720

        # drawing rectangles
        self.ix = 0
        self.endx = 0
        self.iy = 0
        self.endy = 0

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
        sqsize = int(math.sqrt(9))  # half height
        sqsize = int(math.sqrt(9))  # half width

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
    
    def ref_capture (self):
        self.imgRef=self.imgIn.copy()
        print("captured ref")

    def get_frame(self):
        if self.vid.isOpened():
            width = self.width
            height = self.height
            ret, frame = self.vid.read()
            if ret:
                iwidth = int(width)
                iheight = int(height)
                frame = self.split_frame(frame)

                self.imgIn = cv2.resize(
                    frame, (iwidth, iheight), interpolation=cv2.INTER_AREA)
                return (ret, cv2.cvtColor(self.imgIn, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (None, None)     

class App:
    def __init__(self, window):
        self.window = window
        self.window.title("FYP PI02")
        
        self.active_quad=0 #set active quadrant

        self.vid = cv2.VideoCapture(1)
        
        self.video_captures = []
        
        size = 9

        self.video_captures.append(MyVideoCapture(0, size))
        self.video_captures.append(MyVideoCapture(1, size))
        self.video_captures.append(MyVideoCapture(2, size))
        self.video_captures.append(MyVideoCapture(3, size))
        self.video_captures.append(MyVideoCapture(4, size))

        self.video_captures.append(MyVideoCapture(5, size))
        self.video_captures.append(MyVideoCapture(6, size))
        self.video_captures.append(MyVideoCapture(7, size))
        self.video_captures.append(MyVideoCapture(8, size))
        
        self.active = 1

        
        #toplevel = tk.Toplevel(window) #'toplevel' can be changed to anything, it is just a variable to hold the top level, 'window' should be whatever variable holds your main window
        #toplevel.title = 'Expanded Quadrant'

        # Use the fixed window size
        self.canvas = tk.Canvas(window, width=1280, height=720)
        self.canvas.pack()

        self.canvas.bind('<Motion>', self.mouseMove)
        self.canvas.bind("<Button-1>", self.startpaint)
        self.canvas.bind("<ButtonRelease-1>", self.endpaint)

        #buttons
        self.btn_clearmask = tk.Button(
            window, text="Clear Mask", width=30, command=self.clearmask)
        self.btn_clearmask.pack(side=tk.RIGHT, anchor=tk.NE, expand=True)
        self.btn_snapshot = tk.Button(
            window, text="Take Reference", width=30, command=self.video_captures[self.active_quad].ref_capture())
        self.btn_snapshot.pack(side=tk.RIGHT, anchor=tk.NE, expand=True)
        
        self.btn_snapshot = tk.Button(
            window, text="Next Page", width=30, command=self.slide())
        self.btn_snapshot.pack(side=tk.RIGHT, anchor=tk.NE, expand=True)
        
        #self.btn_snapshot = tk.Button(
            #window, text="Enlarge Quadrant", width=30, command=self.ubt1())
        #self.btn_snapshot.pack(side=tk.RIGHT, anchor=tk.NE, expand=True)
        

        #quad buttons
        self.btn_Q1 = tk.Button(
            window, text="Q1", width=15, relief="raised", command= lambda: self.toggle(1))
        self.btn_Q2 = tk.Button(
            window, text="Q2", width=15, relief="raised", command= lambda: self.toggle(2))
        self.btn_Q3 = tk.Button(
            window, text="Q3", width=15, relief="raised", command= lambda: self.toggle(3))
        self.btn_Q4 = tk.Button(
            window, text="Q4", width=15, relief="raised", command= lambda: self.toggle(4))
        
        time.sleep(2)
        
        self.btn_Q1.pack(anchor=tk.CENTER, expand=True)
        self.btn_Q2.pack(anchor=tk.CENTER, expand=True)
        self.btn_Q3.pack(anchor=tk.CENTER, expand=True)
        self.btn_Q4.pack(anchor=tk.CENTER, expand=True)
        
        # after called once, update auto called
        self.delay = 15
        self.update()

        self.window.mainloop()

    def toggle(self, quad):
        if quad==1:
            self.btn_Q1.config(relief="sunken")
            self.btn_Q2.config(relief="raised")
            self.btn_Q3.config(relief="raised")
            self.btn_Q4.config(relief="raised")
        elif quad==2:
            self.btn_Q1.config(relief="raised")
            self.btn_Q2.config(relief="sunken")
            self.btn_Q3.config(relief="raised")
            self.btn_Q4.config(relief="raised")
        elif quad==3:
            self.btn_Q1.config(relief="raised")
            self.btn_Q2.config(relief="raised")
            self.btn_Q3.config(relief="sunken")
            self.btn_Q4.config(relief="raised")
        elif quad==4:
            self.btn_Q1.config(relief="raised")
            self.btn_Q2.config(relief="raised")
            self.btn_Q3.config(relief="raised")
            self.btn_Q4.config(relief="sunken")
        
        self.active_quad=quad-1 #0 indexed array of video_captures vs 1 indexed quad in this function
        print(self.active_quad)

    def clearmask(self):
        self.video_captures[self.active_quad].imgMask.fill(0)
        self.video_captures[self.active_quad].imgTemp.fill(0)
        self.video_captures[self.active_quad].imgInDraw.fill(0)
    
    def slide(self):
        self.active = 1
        
        '''
        size = len(self.video_captures)
        if self.active<=size/4:
            self.active+=1
        else: self.active = 1
        print ("active:", self.active)
        '''

    def mouseMove(self, e):
        x = e.x
        y = e.y
        print("Mouse: ", x, y)

    def startpaint(self, event):
        self.video_captures[self.active_quad].iy
        self.video_captures[self.active_quad].iy, self.video_captures[self.active_quad].ix, self.video_captures[self.active_quad].endx, self.video_captures[self.active_quad].endy

        print("AHHHHHHHHHHHHHHHHHHHHHHHHH", self.video_captures[self.active_quad].ix,
              self.video_captures[self.active_quad].iy, self.video_captures[self.active_quad].endx, self.video_captures[self.active_quad].endy)
        self.drawing = True
        self.video_captures[self.active_quad].ix, self.video_captures[self.active_quad].iy = (
            event.x, event.y)

    def endpaint(self, event):
        self.video_captures[self.active_quad].endx, self.video_captures[self.active_quad].endy

        print("ENDDDDDDDDDDDDDDDDDDDD",
              self.video_captures[self.active_quad].endx, self.video_captures[self.active_quad].endy)
        self.drawing = False
        self.video_captures[self.active_quad].endx, self.video_captures[self.active_quad].endy = (event.x, event.y)

    def update(self):
        image_frames = []
        
        if self.active<=(len(self.video_captures)/4):
            vids = self.video_captures[((self.active-1)*4):((self.active-1)*4+4)]
        else:
            vids = self.video_captures[((self.active-1)*4):]
        print("image length:", len(vids))
        
        for video in vids:
            ret, frame = video.get_frame()
            if ret:
                print("img", type(frame), "frame num:", video.quad_num)
                image_frames.append(cv2.resize(frame, (640,360)))
                
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
            row_images = images[row_index * grid_size: (row_index + 1) * grid_size]
            concatenated_row = cv2.hconcat(row_images)
            if merged_image is None:
                merged_image = concatenated_row
            else:
                merged_image = cv2.vconcat([merged_image, concatenated_row])
    
        return merged_image
   
    def align_images(self, images):
        width = max(image.shape[1] for image in images)
        height = max(image.shape[0] for image in images)
        aligned_images = [cv2.resize(image, (width, height)) for image in images]
        return aligned_images


def main():
    # num_cameras = 4  # Change this variable to the number of available cameras
    root = tk.Tk()
    app = App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
