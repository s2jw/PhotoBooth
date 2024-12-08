import cv2
import os
import smtplib
import imghdr
import numpy as np
from ultralytics import YOLO
from email.message import EmailMessage
import PIL.Image, PIL.ImageTk
from tkinter import *

SMTP_SERVER = 'smtp.naver.com'
SMTP_PORT = 465
EMAIL_ADDR = 's2jw_@naver.com'
EMAIL_PASSWORD = 'godqhrgkwk!!'

class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source

        # Open video source
        self.vid = MyVideoCapture(self.video_source)

        # Create a canvas that can fit the above video source size
        self.canvas = Canvas(window, width=self.vid.width, height=self.vid.height)
        self.canvas.pack()

        # Button 
        self.btn_toggle = Button(window, text="grayscale", width=10, command=self.grayscale)
        self.btn_toggle.pack(side=LEFT, expand=True)
        # self.btn_toggle.grid(row=1, column=5)

        self.btn_toggle = Button(window, text="hsv", width=10, command=self.hsv)
        self.btn_toggle.pack(side=LEFT, expand=True)

        self.btn_toggle = Button(window, text="reverse", width=10, command=self.reverse)
        self.btn_toggle.pack(side=LEFT, expand=True)

        self.btn_toggle = Button(window, text="edge", width=10, command=self.edge)
        self.btn_toggle.pack(side=LEFT, expand=True)

        self.btn_toggle = Button(window, text="heart flare", width=10, fg="pink", command=self.heartFlare)
        self.btn_toggle.pack(side=LEFT, expand=True)

        self.btn_toggle = Button(window, text="dot heart", width=10, fg="pink", command=self.dotHeart)
        self.btn_toggle.pack(side=LEFT, expand=True)

        self.btn_toggle = Button(window, text="confetti", width=10, fg="pink", command=self.confetti)
        self.btn_toggle.pack(side=LEFT, expand=True)

        # enter your email 
        self.email_label = Label(window, text="Enter your email : ")
        self.email_label.pack(side=TOP, expand=True)

        self.email_entry = Entry(window, width=30)
        self.email_entry.pack(side=TOP, expand=True)

        self.submit_button = Button(window, text="submit", width=10, fg="red", command=self.submit_email)
        self.submit_button.pack(side=BOTTOM, expand=True)

        self.window.bind('<space>', self.capture)

        self.grayscale = False
        self.hsv = False
        self.reverse = False
        self.edge = False
        self.heartFlare = False
        self.dotHeart = False
        self.confetti = False

        self.overlay_images = ''
        self.capNum = 0
        self.delay = 15
        self.update()
        
        self.window.mainloop()

    def grayscale(self):
        self.grayscale = not self.grayscale
        self.hsv = False
        self.reverse = False
        self.edge = False
        self.heartFlare = False

    def hsv(self):
        self.hsv = not self.hsv
        self.grayscale = False
        self.reverse = False
        self.edge = False
        self.heartFlare = False

    def reverse(self):
        self.reverse = not self.reverse
        self.grayscale = False
        self.hsv = False
        self.edge = False
        self.heartFlare = False

    def edge(self):
        self.edge = not self.edge
        self.grayscale = False
        self.hsv = False
        self.reverse = False
        self.heartFlare = False

    def heartFlare(self):
        self.heartFlare = not self.heartFlare
        self.edge = False
        self.grayscale = False
        self.hsv = False
        self.reverse = False
        self.dotHeart = False

        self.yolo = YOLO('yolov8n-pose.pt')
        self.images_path1 = './heartFlare'  # 이미지 디렉토리 경로
        self.overlay_images1 = self.load_images(self.images_path1)
        self.image_index1 = 0

    def dotHeart(self):
        self.dotHeart = not self.dotHeart
        self.edge = False
        self.grayscale = False
        self.hsv = False
        self.reverse = False
        self.heartFlare = False

        self.yolo = YOLO('yolov8n-pose.pt')
        self.images_path2 = './dotHeart'  # 이미지 디렉토리 경로
        self.overlay_images2 = self.load_images(self.images_path2)
        self.image_index2 = 0

    def confetti(self):
        self.confetti = not self.confetti
        self.edge = False
        self.grayscale = False
        self.hsv = False
        self.reverse = False
        self.heartFlare = False
        self.dotHeart = False

        self.yolo = YOLO('yolov8n-pose.pt')
        self.images_path3 = './confetti'  # 이미지 디렉토리 경로
        self.overlay_images3 = self.load_images(self.images_path3)
        self.image_index3 = 0

    def load_images(self, images_path):
        # 디렉토리에서 모든 PNG 이미지 로드
        filenames = [f for f in os.listdir(images_path) if f.endswith('.png')]
        images = [cv2.imread(os.path.join(images_path, f), cv2.IMREAD_UNCHANGED) for f in filenames]
        images = [cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA) if img.shape[2] == 4 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]
        return images

    def capture(self, event=None):
        ret, frame = self.vid.get_frame()

        if ret:
            frame = cv2.flip(frame, 1)
            if self.grayscale:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            elif self.hsv:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            elif self.reverse:
                frame = cv2.bitwise_not(frame)
            elif self.edge:
                frame = cv2.Canny(frame, 100, 255)

            cv2.imwrite('/Users/choijeong-won/Documents/VsCode/photobooth/capture/%03d.png' % self.capNum, frame)
            self.capNum += 1
            print("Captured!")

    def overlay_image_onto_frame(self, frame, img, x, y, width, height):
        # 오버레이 이미지의 크기 조정
        img = cv2.resize(img, (width, height))

        # 오버레이
        try:
            for c in range(0, 3):
                frame[y:y+height, x:x+width, c] = img[:, :, c] * (img[:, :, 3]/255.0) +  frame[y:y+height, x:x+width, c] * (1.0 - img[:, :, 3]/255.0)
        except:
            pass

        return frame

    def submit_email(self):
        your_email = self.email_entry.get()
        if not your_email:
            print("No email entered!")
            return

        try:
            self.send_mail(your_email)
            print(f"Email sent to {your_email}")
        except Exception as e:
            print(f"Failed to send email: {e}")
            
    def send_mail(self, email):
        smtp = smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT)
        #smtp.starttls()
        smtp.login(EMAIL_ADDR, EMAIL_PASSWORD)

        msg = EmailMessage()
        msg.set_content('This is your Photo Booth pictures! Have a good day :)')
        msg["Subject"] = "Photo Booth Capture"
        msg["From"] = EMAIL_ADDR
        msg["To"] = email

        for i in range(self.capNum):
            with open(f'/Users/choijeong-won/Documents/VsCode/photobooth/capture/{i:03d}.png', 'rb') as image:
                image_file = image.read()  # 이미지 파일 읽어오기
                image_type = imghdr.what(None, image_file) or 'png'
                msg.add_attachment(image_file, maintype='image', subtype=image_type, filename=f'capture_{i:03d}.png')

        smtp.send_message(msg)
        smtp.quit()

    def overlay(self, overlay_images, image_index):
        results = self.yolo(frame, stream=True)
        for result in results:
            boxes = result.boxes

            for box in boxes:
                if box.cls != 0 or box.conf < 0.5:  # 사람 클래스의 박스만 처리
                    continue
                    
                x, y, w, h = box.xywh[0]  # 중심 좌표와 너비, 높이 추출
                x1 = int(x - w / 2)  # 박스 시작 좌표 계산
                y1 = int(y - h / 2)

                # 오버레이할 이미지를 박스 크기에 맞게 조정하여 프레임에 적용
                frame = self.overlay_image_onto_frame(frame, self.overlay_images[self.image_index], x1, y1, int(w), int(h))
            self.image_index = (self.image_index + 1) % len(self.overlay_images)

    def update(self):
        ret, frame = self.vid.get_frame()

        if ret:
            if self.grayscale:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            elif self.hsv:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            elif self.reverse:
                frame = cv2.bitwise_not(frame)
            elif self.edge:
                frame = cv2.Canny(frame, 100, 255)
            elif self.heartFlare:
                #self.overlay(self.overlay_images1, self.image_index1)
                results = self.yolo(frame, stream=True)
                for result in results:
                    boxes = result.boxes

                    for box in boxes:
                        if box.cls != 0 or box.conf < 0.5:  # 사람 클래스의 박스만 처리
                            continue
                            
                        x, y, w, h = box.xywh[0]  # 중심 좌표와 너비, 높이 추출
                        x1 = int(x - w / 2)  # 박스 시작 좌표 계산
                        y1 = int(y - h / 2)

                        # 오버레이할 이미지를 박스 크기에 맞게 조정하여 프레임에 적용
                        frame = self.overlay_image_onto_frame(frame, self.overlay_images1[self.image_index1], x1, y1, int(w), int(h))
                    self.image_index1 = (self.image_index1 + 1) % len(self.overlay_images1)

            elif self.dotHeart:
                results = self.yolo(frame, stream=True)
                for result in results:
                    boxes = result.boxes

                    for box in boxes:
                        if box.cls != 0 or box.conf < 0.5:  # 사람 클래스의 박스만 처리
                            continue
                            
                        x, y, w, h = box.xywh[0]  # 중심 좌표와 너비, 높이 추출
                        x1 = int(x - w / 2)  # 박스 시작 좌표 계산
                        y1 = int(y - h / 2)

                        # 오버레이할 이미지를 박스 크기에 맞게 조정하여 프레임에 적용
                        frame = self.overlay_image_onto_frame(frame, self.overlay_images2[self.image_index2], x1, y1, int(w), int(h))
                    self.image_index2 = (self.image_index2 + 1) % len(self.overlay_images2)

            elif self.confetti:
                results = self.yolo(frame, stream=True)
                for result in results:
                    boxes = result.boxes

                    for box in boxes:
                        if box.cls != 0 or box.conf < 0.5:  # 사람 클래스의 박스만 처리
                            continue
                            
                        x, y, w, h = box.xywh[0]  # 중심 좌표와 너비, 높이 추출
                        x1 = int(x - w / 2)  # 박스 시작 좌표 계산
                        y1 = int(y - h / 2)

                        # 오버레이할 이미지를 박스 크기에 맞게 조정하여 프레임에 적용
                        frame = self.overlay_image_onto_frame(frame, self.overlay_images3[self.image_index3], x1, y1, int(w), int(h))
                    self.image_index3 = (self.image_index3 + 1) % len(self.overlay_images3)

            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = NW)

        self.window.after(self.delay, self.update)

class MyVideoCapture:
    def __init__(self, video_source=0):
        capNum = 0
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

# Create a window and pass it to the Application object
App(Tk(), "Photo Booth")
