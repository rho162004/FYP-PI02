import tkinter as tk
from tkinter import messagebox
from tkinter import *
import cv2
import datetime
from PIL import Image as Img
from PIL import ImageTk
import numpy as np
import requests
import cv2
from cryptography.fernet import Fernet
import math

#make it automatically go to the next camera / link to arrows / stop and go

'''
screenshot.txt is a necessary document required to load settings on startup
screenshot.txt contains all saved vars in this order:
    0. Screenshot number (photonum)
    1. Click option ()
    2. Click2 option (secondselect)
    3. timer setting (seconds)
    4. token (token)
    5. chat id (chat_id)
'''
#Declare variable to read from the text file
j = open('screenshot.txt', 'r')

#Variable read from declared variable and collect the data in it
vars = j.readlines()
token = vars[4].strip()
chat_id = vars[5].strip()
secondselect = vars[3].strip()
photonum = vars[0].strip()

#Close an open file so memory does not get leaked and slow/crash program
j.close()

#For sendidng message and image to TELEGRAM
def send_msg(text):
   global chat_id, token
   url_req = "https://api.telegram.org/bot" + token + "/sendMessage" + "?chat_id=" + chat_id + "&text=" + text 
   results = requests.get(url_req)

def send_img():
    global chat_id, token
    files ={'photo':open('danger.png', 'rb')}
    url_req = "https://api.telegram.org/bot" + token + "/sendPhoto" + "?chat_id=" + chat_id
    resp = requests.post(url_req ,files=files)

#For security passwords/login
def generate_key():
    return Fernet.generate_key()

def encrypt_password(key, password):
    f = Fernet(key)
    return f.encrypt(password.encode()).decode()

def decrypt_password(key, encrypted_password):
    f = Fernet(key)
    return f.decrypt(encrypted_password.encode()).decode()
passwords = {}

#Login window class
class Application:
    def __init__(self, window):   
        self.key = generate_key()

        instructions = '''Please enter username and password for security reason'''

        #creates window
        self.window=window
        self.window.title("Security Login")
        self.window.configure(bg="grey")
        self.window.resizable(False, False)

        #frame that all components exist on
        center_frame = tk.Frame(self.window, bg="#d3d3d3")
        #grid function says where to place an item in the 'grid' on the window
        center_frame.grid(row=0, column=0, padx=10, pady=10)

        instruction_label = tk.Label(center_frame, text=instructions, bg="#d3d3d3")
        instruction_label.grid(row=0, column=1, padx=10, pady=5)

        service_label = tk.Label(center_frame, text="Username:", bg="#d3d3d3")
        service_label.grid(row=1, column=0, padx=10, pady=5)
        self.service_entry = tk.Entry(center_frame)
        self.service_entry.grid(row=1, column=1, padx=10, pady=5)

        password_label = tk.Label(center_frame, text="Password:", bg="#d3d3d3")
        password_label.grid(row=3, column=0, padx=10, pady=5)
        self.password_entry = tk.Entry(center_frame, show="*")
        self.password_entry.grid(row=3, column=1, padx=10, pady=5)

        add_button = tk.Button(center_frame, text="Login", command=self.login, height=1, width=10)
        add_button.grid(row=5, column=4, padx=10, pady=5)
        self.window.mainloop()

    #fetch values and test against set password/username
    def login(self):
        service = self.service_entry.get()
        #username = username_entry.get()
        password = self.password_entry.get()
        admin_pass = 'admin'
        admin_login = 'admin'

        if service and password:
            self.encrypted_password = encrypt_password(self.key, password)
            passwords[service] = {'username': service, 'password': self.encrypted_password}

            if service == admin_login and password == admin_pass:
                #close all
                self.window.destroy()
                send_msg("Security access granted. Login authenticated.")
                switch()                                                      #what what what what
            else:
                messagebox.showwarning("Error", "Invalid login and password.") 


        else:
            messagebox.showwarning("Error", "Please fill in all the fields.")

#main GUI pass
class App:
    #Initialize
    def __init__(self, window, video_source=0):
        global vars, num_cams #globals
        
        #from number of cams, find max page number
        self.num_cams = num_cams
        if self.num_cams%4 == 0:
            self.max_page=self.num_cams/4
        else:
            self.max_page = self.num_cams/4 +1

        #initialize all cameras
        self.video_captures = []
        for camera in range(num_cams):
            self.video_captures.append(MyVideoCapture(camera))
            print("Camera #"+str(camera)+' added')
        self.active_quad = 0

        #Create tkinter window
        self.window = window
        self.window.title("Security Camera")
        self.window.geometry("1200x800") 
        self.window.resizable(False, False)
        
        #frame for image
        self.frame = tk.Frame(self.window, width = 200, height = 200)
        self.frame.grid(row = 1, column = 0, sticky = W+N)

        #frame for buttons
        self.buttonframeleft = tk.Frame(self.window)
        self.buttonframeleft.grid(row = 2, column = 2, sticky = E+N)
        self.buttonframeright = tk.Frame(self.window)
        self.buttonframeright.grid(row = 2, column = 2, sticky = W+N)

        # create a canvas that can fit the video source size
        self.window.grid_rowconfigure(list(range(10)), minsize=5)
        self.window.grid_columnconfigure(list(range(10)), minsize=60)
        self.window.grid_columnconfigure(1, minsize=100)

        self.window.grid_propagate(False)
        self.frame.grid_propagate(False)
        self.mouseDown = False

        #canvas that the camera lies on
        self.canvas = tk.Canvas(window, width=self.video_captures[self.active_quad].width*2, height=self.video_captures[self.active_quad].height*3)
        self.canvas.pack()
        #self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.toggleMouse)
        self.canvas.bind("<Button-1>", self.toggleMouse)

        #buttons ! 
        self.btn_snapshot = tk.Button(self.buttonframeleft, text="snapshot", width=30, command=self.video_captures[self.active_quad].snapshot)
        self.btn_clearmask = tk.Button(self.buttonframeright, text="Clear Mask", width=30, command=self.clearmask)
        
        self.canvas.grid(row = 1, column = 2, sticky = W, rowspan=3) 
        self.btn_snapshot.grid(row = 2, column = 5, sticky = W+N)     
        self.btn_clearmask.grid(row = 2, column = 1, sticky = W+N)
        
        #For Time
        self.slidertime = Scale(self.frame, from_=0, to=50, orient=HORIZONTAL, command=self.changeseconds)
        self.defaulttime = Button( self.frame , text = "Default", command=self.default_time, height=1, width=8)

        #For Detection
        self.click2 = StringVar()
        self.click2.set(vars[2].strip())
        detect2 = [
            "Auto Detect", 
            "Select AOI", 
        ]
        self.drop2 = OptionMenu(self.frame, self.click2, *detect2, command=self.secondselect)

        #For Threshold
        self.sliderthres = Scale(self.frame, from_=0, to=50, orient=HORIZONTAL)
        self.defaultthres = Button( self.frame , text = "Default", command=self.default_thres, height=1, width=8)

        #Dropdown menu options
        detect_options = [
            "Main Menu",
            "Detection",
            "Time", 
            "Change Telegram Channel",
            "Log Out",
        ]

        #Datatype of menu text
        self.clicked = StringVar()

        #Initial Menu Text
        self.clicked.set(vars[1].strip())

        #Create Drop Down Menu
        drop = OptionMenu(self.window, self.clicked, *detect_options, command=self.selected)
        drop.grid(row = 0, column = 0, sticky = W+N)

        #Create Label 
        label = Label(self.window , text = " " ) 
        label.grid(row = 0, column = 0, sticky = W) 

        #Create Play and Stop button
        resizedplayphoto = cv2.imread('play.png')
        resizedplayphoto = cv2.resize(resizedplayphoto, (20,20), interpolation=cv2.INTER_AREA)
        playphoto = Img.fromarray(resizedplayphoto)
        playphoto = ImageTk.PhotoImage(image=playphoto)

        resizedstopphoto = cv2.imread('stop.png')
        resizedstopphoto = cv2.resize(resizedstopphoto, (20,20), interpolation=cv2.INTER_AREA)
        stopphoto = Img.fromarray(resizedstopphoto)
        stopphoto = ImageTk.PhotoImage(image=stopphoto)

        playbutton = Button(self.buttonframeleft, text = 'Click Me!', image = playphoto, command=self.selected)
        stopbutton = Button(self.buttonframeright, text = 'Click Me!', image = stopphoto, command=self.selected)
        previouspage = Button(self.buttonframeleft, text = 'Previous', command=self.prevpage)
        nextpage = Button(self.buttonframeright, text = 'Next', command=self.nextpage)

        playbutton.grid(row = 2, column = 3, sticky = W+N)
        stopbutton.grid(row = 2, column = 5, sticky = E+N)
        previouspage.grid(row = 2, column = 4, sticky = E+N)
        nextpage.grid(row = 2, column = 4, sticky = W+N)

        #telegram msging setup
        self.token_entry = tk.Entry(self.frame)
        self.chat_id_entry = tk.Entry(self.frame)
        self.token_label = tk.Label(self.frame,text = "Telegram Token:")
        self.chat_id_label = tk.Label(self.frame, text = "Telegram Chat_ID:")
        self.newchatid = tk.Button(self.frame,text="Change ChatID", command=self.updateid)
        self.newtoken = tk.Button(self.frame,text="Change Token", command=self.updatetoken)

        #dropdown and button settings
        self.selected("")
        self.change = False
        self.page_num=1

        # after called once, update auto called
        self.delay = 15
        self.update()
        # Execute tkinter 
        self.window.mainloop() 

    #go back a page in cameras
    def prevpage(self,e):
        if self.page_num!=1:
            self.page_num-=1
            print("At page", self.page_num)
        else:
            print("Already at first page!")

    #go forward a page in cameras
    def nextpage(self):
        if self.page_num!=self.max_page:
            self.page_num+=1
            print("At page", self.page_num)
        else:
            print("Already at last page!")

    #change time value
    def changeseconds(self, e):
        #read val off of slider and set as self.vid.seconds
        global seconds
        seconds = int(e)
        print("Timer length:", e, "seconds")

    #second dropdown menu get the selected option
    def secondselect(self, event):
        global secondselect
        secondselect = self.click2.get()
    
    #find mouse location n paint
    def toggleMouse(self, e):
        if not self.mouseDown:
            self.mouseDown = True
            print("clicking")
            self.video_captures[self.active_quad].ix = e.x
            self.video_captures[self.active_quad].iy = e.y
            self.mouseOri = (e.x, e.y)
            print("ix, iy:", self.video_captures[self.active_quad].ix, self.video_captures[self.active_quad].iy)
        else:
            self.mouseDown = False
            self.mousePt = (e.x, e.y)
            print("mousept:", self.mousePt)
            print("releasing")
            self.paint()

    #runs only on close/destruction of App
    def __del__(self):
        global vars, j, chat_id, token, seconds, photonum
        j.close()        
        words = str(photonum)+"\n"+self.clicked.get()+"\n"+ str(self.click2.get())+"\n"+ str(seconds)+"\n"+ token+"\n"+ chat_id+"\n"
        print(words)
        l = open("screenshot.txt", "w")
        l.write(words)

    #main dropdown menu get option
    def selected(self, event):
        global vars
        if self.clicked.get() == 'Detection':
            self.defaulttime.grid_forget()
            self.slidertime.grid_forget()
            self.token_entry.grid_forget()
            self.token_label.grid_forget()
            self.chat_id_entry.grid_forget()
            self.chat_id_label.grid_forget()
            self.newchatid.grid_forget()
            self.newtoken.grid_forget()

            self.secondselect("")

            self.drop2.grid(row=1, column=0, sticky=W+N)

        elif self.clicked.get() == 'Time':
            self.drop2.grid_forget()
            self.token_entry.grid_forget()
            self.token_label.grid_forget()
            self.chat_id_entry.grid_forget()
            self.chat_id_label.grid_forget()
            self.newchatid.grid_forget()
            self.newtoken.grid_forget()

            self.slidertime.set(int(vars[3].strip()))
            self.slidertime.grid(row = 2, column = 0, sticky = W+N)
            self.defaulttime.grid(row = 1, column = 0, sticky = N+W)

        elif self.clicked.get() == 'Log Out':
            self.window.destroy()                               #what what what what what
            switch()

        elif self.clicked.get() == "Main Menu":
            self.drop2.grid_forget()
            self.slidertime.grid_forget()
            self.defaulttime.grid_forget()
            self.token_entry.grid_forget()
            self.token_label.grid_forget()
            self.chat_id_entry.grid_forget()
            self.chat_id_label.grid_forget()
            self.newchatid.grid_forget()
            self.newtoken.grid_forget()
            
        elif self.clicked.get() == "Change Telegram Channel":
            global chat_id, token
            self.slidertime.grid_forget()
            self.defaulttime.grid_forget()
            self.drop2.grid_forget()

            self.token_label.grid(row=1, column=0, sticky=W+N)
            self.token_entry.grid(row=1,column=1, sticky = W+N)
            self.chat_id_label.grid(row=2,column=0, sticky = W+N)
            self.chat_id_entry.grid(row=2,column=1, sticky = W+N)  
            self.newchatid.grid(row=3, column=0, sticky = W+N)
            self.newtoken.grid(row=3, column=1, sticky= W+N)

        else:
            self.drop2.grid_forget()
            self.slidertime.grid_forget()
            self.defaulttime.grid_forget()
            self.token_entry.grid_forget()
            self.token_label.grid_forget()
            self.chat_id_entry.grid_forget()
            self.chat_id_label.grid_forget()
            self.newchatid.grid_forget()
            self.newtoken.grid_forget()
    
    #telegram token and chat id update
    def updatetoken(self):
        global token   
        #get the info
        token = self.token_entry.get()

        info = token
        print(info)

        print("Telegram token changed")
        self.change = True
    
    def updateid(self):
        global chat_id   
        #get the info
        chat_id = self.chat_id_entry.get()

        info = chat_id
        print(info)
        j = open("telegramid.txt", "w")
        j.write(info)

        print("Telegram chatID changed")

    #default button for timer
    def default_time(self):
        self.slidertime.set(5)

    #painting by getting mouse location n drawing a rectangle
    def paint(self):
        self.video_captures[self.active_quad].imgMask = self.video_captures[self.active_quad].imgMask.copy()

        cv2.rectangle(self.video_captures[self.active_quad].imgMask, self.mouseOri, self.mousePt, (255, 255, 255), -1)

        print("rectangle from", self.video_captures[self.active_quad].ix, self.video_captures[self.active_quad].iy, self.mousePt)

    #delete the rectangle
    def clearmask(self):
        self.video_captures[self.active_quad].clearmask()

    #set threshold back to default
    def default_thres(self):
        self.sliderthres.set(50)

    #update runs every self.delay amount of ms
    def update(self):
        page_vids = []
        page_vids = self.video_captures[(self.page_num*4-4):(self.page_num*4-1)]

        image_frames = []
        
        for video in self.video_captures:
            ret, frame = video.get_frame()
            if ret:
                image_frames.append(frame)
            else: 
                print("image cannot load", str(video.quad_num))
                
        merged_image = self.assemble_grid(image_frames)
        merged_image = Img.fromarray(merged_image)
        
        self.photo = ImageTk.PhotoImage(image=merged_image)
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
    
        self.window.after(self.delay, self.update)
    
    #put images together
    def concat_vh(self, list_2d):
        # return final image
        return cv2.vconcat([cv2.hconcat(list_h) 
                        for list_h in list_2d])
    
    #put images together
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

#main camera class
class MyVideoCapture:
    def __init__(self, quad_num, video_source=0):
        global vars, num_cams
        # Open the video source
        self.num_cams = num_cams
        self.quad_num = quad_num

        #get video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)
        
        #variable setup
        self.mousePt = (0, 0)
        self.runonce = True
        self.timer = datetime.datetime(2027, 1, 1, 1, 1, 1, 1)
        self.current = datetime.datetime.now()
        self.timesup = False
        self.dangerous = False
        
        self.click2 = "Select Option"
        self.seconds = 5

        #set camera res
        w = 1280.0/4
        h = 720.0/4
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        print("res set: ", w, h)

        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print("final res: ", self.width, self.height)

        #image setups
        self.imgRef = np.zeros(
            (int(self.height), int(self.width), 3), dtype="uint8")
        self.imgComposite = np.zeros((int(self.height), int(
            self.width), 3), dtype="uint8")   # blank images
        self.imgTemp1 = np.zeros(
            (int(self.height), int(self.width), 3), dtype="uint8")
        self.imgBackground = np.zeros(
            (int(self.height), int(self.width), 3), dtype="uint8")
        self.imgMask = np.zeros(
            (int(self.height), int(self.width), 3), dtype="uint8")

        # Copy an image of snapshot for reference
        self.imgMask = self.imgRef.copy()
        self.phototaken = False
        self.danger = False

    #save a picture to files
    def snapshot(self):
        global photonum
        self.imgRef = self.imgIn.copy()
        title = str(self.quad_num)+"screenshot"+str(self.photonum)+".png"
        cv2.imwrite(title, self.imgRef)

    def __del__(self):
        # Release the video source when the object is destroyed
        if self.vid.isOpened():
            self.vid.release()
            cv2.destroyAllWindows()
            print("Stream ended")

    #put image together
    def concat_vh(self, list_2d):
        # return the final image
        return cv2.vconcat([cv2.hconcat(list_h) for list_h in list_2d])
    
    #split the incoming frame into seperate video streams
    def split_frame(self, merged_frame):
        height, width, _ = merged_frame.shape
        #print("shape h:", height, "shape w:", width)

        # assuming that the number of squares is a square number (height and width of grid is same number of images)
        sqsize = int(math.sqrt(self.num_cams))  # half height
        sqsize = int(math.sqrt(self.num_cams))  # half width

        col = int(self.quad_num % sqsize)
        row = int(self.quad_num / sqsize)
        self.quad_num
        
        hheight = height/sqsize
        hwidth = width/sqsize
        
        r1 = int((hheight*row))
        r2 = int(hheight*(row+1))
        c1 = int((hwidth*col))
        c2 = int(hwidth*(col+1))
        
        quad = merged_frame[r1:r2, c1:c2]

        return quad

    #outputs a frame to be put in App
    def get_frame(self):
        global seconds
        self.seconds = seconds
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # Detect Objects

                # Load yolo
                net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

                classes = []
                with open("coco.names", 'r') as f:
                    classes = [line.strip() for line in f.readlines()]
                # print(classes)
                layer_name = net.getLayerNames()
                output_layer = [layer_name[i - 1] for i in net.getUnconnectedOutLayers()]
                colors = np.random.uniform(0, 255, size=(len(classes), 3))

                #split frame and drawing
                drawing = False
                self.ix, self.iy = -1, -1
                self.imgIn = self.split_frame(frame)
                if self.click2 == "Select AOI":
                    frame = cv2.bitwise_and(self.imgIn, self.imgMask)
                
                #more yolo
                blob = cv2.dnn.blobFromImage(
                    frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
                net.setInput(blob)
                outs = net.forward(output_layer)
                
                print("self.click2", self.click2)

                # more more yolo
                class_ids = []
                confidences = []
                boxes = []
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
                            # cv.circle(img, (center_x, center_y), 10, (0, 255, 0), 2 )
                            # Rectangle Coordinate
                            x = int(center_x - w/2)
                            y = int(center_y - h/2)
                            boxes.append([x, y, w, h])
                            confidences.append(float(confidence))
                            class_ids.append(class_id)
                indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
                print(indexes)
        
                font = cv2.FONT_HERSHEY_SIMPLEX
                presence = ""
                count = 0
                danger = False
                for i in range(len(boxes)):
                    if i in indexes:
                        x, y, w, h = boxes[i]
                        cx = int((x + x + w) / 2)
                        cy = int((y + y + h) / 2)
                        label = str(classes[class_ids[i]])
                        print(label)
                        confidence = f'{confidences[i]:.3f}'

                        #simple rules for ladder detection, if one person in frame then alert after seconds amount of time
                        if ((presence == "person" and label == "cell phone") or (presence == "cell phone" and label == "person")):
                            if count == 1:
                                print("Test Succeeded - Detecting One Person")
                                self.danger = True
                            elif count == 2:
                                print("Test Succeeded - Detecting Two People")
                                self.danger = False

                        if label == "cell phone" or label == "person":
                            presence = label

                        if label == "person":
                            count += 1
                        
                        if self.danger == True and label == "cell phone":
                            self.current = datetime.datetime.now() 
                            if (self.timesup == False) and self.runonce == True:
                                
                                print("Current date:", str(self.current))
                                self.timer = self.current + datetime.timedelta(seconds=self.seconds)
                                print("timer time is:", self.timer)
                                print("Time in", str(self.seconds),  "seconds:", str(self.timer))
                                self.runonce = False
                            if self.current >= self.timer:
                                
                                if self.timesup == True:    
                                    print("time up")
                                    color = colors[i]
                                    cv2.rectangle(
                                        frame, (x, y), (x + w, y + h), color, 1)
                                    cv2.putText(frame, label + ' : ' + confidence + ' people count: ' +
                                                str(count), (x + 2, y + 10), font, 0.3, color, 1)
                                else:
                                    cv2.imwrite("danger.png", frame)
                                    #writer.write(frame)
                                    send_msg("Danger Detected:")
                                    send_img()
                                    self.phototaken = True
                                self.timesup = True
                print("dangerous:", self.dangerous, "danger:", self.danger, "runonce:", self.runonce, "timesup:", self.timesup)
                if self.timesup and not(self.dangerous or self.danger or self. runonce):
                    self.timesup = False
                    self.runonce = True
                    print("resetting timer")
                    #writer.release()
                    print("freedom")

                #convert image to rgb since its otherwise blue
                self.dangerous = self.danger
                self.danger = False
                imgIn = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                imgIn = cv2.resize(imgIn, (int(self.width), int(
                    self.height)), interpolation=cv2.INTER_AREA)

                self.imgOut = imgIn.copy()

                self.imgTemp1 = self.imgMask.copy()
                self.imgTemp1[:, :, 0] = 0

                cv2.addWeighted(imgIn, 0.5, self.imgTemp1,
                                0.5, 0.0, self.imgTemp1)
                #cv2.imshow('imgTemp1', self.imgTemp1)

                self.imgOut = cv2.bitwise_and(imgIn, cv2.bitwise_not(
                    self.imgMask)) + cv2.bitwise_and(self.imgTemp1, self.imgMask)
                #cv2.imshow('imgOut', self.imgOut)

                #cv2.imshow('imgOut', self.imgOut)
                #cv2.imshow('imgMask', self.imgMask)

                # Layout and Display Results in One Window
                self.imgIn = cv2.cvtColor(self.imgIn, cv2.COLOR_BGR2RGB)
                #imgResults = self.concat_vh([[self.imgIn, self.imgOut],
                                             #[imgIn, self.imgOut]])

                return (ret, self.imgIn)
            else:
                return (ret, None)
        else:
            return (ret, None)

#change from Application (login) to App (main gui)
def switch():
    global loggedin, window
    if loggedin:
        loggedin=False
        window=tk.Tk()
        app=Application(window)
    else:
        loggedin=True
        window=tk.Tk()
        app=App(window)

# Create a window and pass it to the Application object
loggedin = False
num_cams = 4
seconds = 5
Application(tk.Tk())


#read text ref: https://www.geeksforgeeks.org/extract-numbers-from-a-text-file-and-add-them-using-python/