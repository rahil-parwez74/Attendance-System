import tkinter as tk
from tkinter import Message, Text
import cv2, os
import shutil
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time
import tkinter.ttk as ttk
import tkinter.font as font

window = tk.Tk()
window.title("Face_Recogniser")
window.geometry('1280x720')
dialog_title = 'QUIT'
dialog_text = 'Are you sure?'
window.configure(background='blue')
window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)

message = tk.Label(window, text="Face Recognition Based Attendance Management System", bg="Green", fg="white", width=50  ,height=3,font=('times', 30, 'italic bold underline'))
message.place(x=50, y=20)

lbl = tk.Label(window, text="Enter ID", width=20, height=2, fg="red", bg="yellow", font=('times', 15, 'bold'))
lbl.place(x=50, y=200)

txt = tk.Entry(window, width=20, bg="yellow", fg="red",  font=('times', 22, 'bold'))
txt.place(x=400, y=215)

lbl2 = tk.Label(window, text="Enter Name", width=20, fg="red", bg="yellow", height=2, font=('times', 15, 'bold'))
lbl2.place(x=50, y=300)

txt2 = tk.Entry(window, width=20, bg="yellow", fg="red", font=('time', 22, 'bold'))
txt2.place(x=400, y=315)

lbl3 = tk.Label(window, text="Notification", width=15, fg="red", bg="yellow", height=2, font=('times', 22, 'bold'))
lbl3.place(x=50, y=400)

message = tk.Label(window, text="", bg="yellow", fg="red", width=30, height=2, activebackground="yellow", font=('times', 15, 'bold'))
message.place(x=400, y=398)

lbl4 = tk.Label(window, text="Attendance", width=15, fg="red", bg="yellow", height=2, font=('times', 22, 'bold underline'))
lbl4.place(x=50, y=620)

message2 = tk.Label(window, text="", bg="yellow", fg="red", width=30, height=2, activebackground="yellow", font=('times', 15, 'bold'))
message2.place(x=400, y=620)


def clear():
    txt.delete(0, 'end')
    res = ""
    message.configure(text=res)

def clear2():
    txt2.delete(0, 'end')
    res = ""
    message.configure(text=res)

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False

def TakeImages():
    Id=(txt.get())
    name=(txt2.get())
    if(is_number(Id) and name.isalpha()):
        cam = cv2.VideoCapture(0)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(harcascadePath)
        sampleNum = 0
        while(True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                sampleNum = sampleNum + 1
                cv2.imwrite("TrainingImage\ "+name +"."+Id + '.'+ str(sampleNum) + ".jpg", gray[y:y+h, x:x+w])
                cv2.imshow('frame',img)

            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            elif sampleNum > 60:
                break

        cam.release()
        cv2.destroyAllWindows()
        res = "Image Save for ID: " + Id +"Name: " + name
        row = [Id, name]
        with open('StudentDetails\studentDetails.csv', 'a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        message.configure(text=res)
    else:
        if(is_number(Id)):
            res = "Enter Alphabetical Name"
            message.configure(text=res)
        if(name.isalpha()):
            res = "Enter Numeric Id"
            message.configure(text=res)

def TrainImage():
    recognizer = cv2.face_LBPHFaceRecognizer.create()
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)
    faces,Id = getImagesAndLabels("TrainingImage") 
    recognizer.train(faces, np.array(Id))
    recognizer.save("TrainingImageLabel\Trainner.yml")
    res = "Image Trained"#+",".join(str(f) for f in id)
    message.configure(text=res)


def getImagesAndLabels(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    faces = []
    Ids = []
    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert('L')
        imageNp=np.array(pilImage)
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(imageNp)
        Ids.append(Id)
    return faces,Ids


def TrackImages():
    recognizer = cv2.face_LBPHFaceRecognizer.create()
    recognizer.read("TrainingImageLabel\Trainner.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath)
    df = pd.read_csv("StudentDetails\studentDetails.csv")
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    col_names = ['Id', 'Name', 'Date', 'Time']
    attendance = pd.DataFrame(columns = col_names)
    while True:
        ret, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)
            Id, conf = recognizer.predict(gray[y:y+h, x:x+w])
            if 'Id' in df and (conf < 50):
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa=df.loc[df['Id'] == Id]['Name'].values
                tt = str(Id)+"-"+aa
                attendance.loc[len(attendance)] = [Id,aa,date,timeStamp]

            else:
                Id = 'Unknown'
                tt = str(Id)
                if(conf > 75):
                    noOfFile = len(os.listdir("ImagesUnknown"))+1
                    cv2.imwrite("ImagesUnknown\Image"+str(noOfFile) + ".jpg", im[y:y+h,x:x+w])
                cv2.putText(im,str(tt),(x,y+h), font, 1,(255,255,255),2)
            attendance= attendance.drop_duplicates(subset=['Id'],keep='first')
            cv2.imshow('im',im)
            if(cv2.waitKey(1)==ord('q')):
                break
        ts = time.time()
        date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
        timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
        Hour,Minute,Second=timeStamp.split(":")
        fileName="Attendance\Attendance_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
        attendance.to_csv(fileName,index=False)
        cam.release()
        cv2.destroyAllWindows()
        res=attendance
        message2.configure(text= res)
    
    
        



clearButton = tk.Button(window, text="Clear", command=clear, fg="red", bg="yellow", height=2, activebackground="Red", font=('times', 22, 'bold'))
clearButton.place(x=950, y=200)

clearButton2 = tk.Button(window, text="Clear", command=clear2, fg="red", bg="yellow", height=2, activebackground="Red", font=('times', 22, 'bold'))
clearButton2.place(x=950, y=300)

takeImg = tk.Button(window, text="Take Image", command=TakeImages, fg="red", bg="yellow", width=20, height=3, activebackground="Red", font=('times', 15, 'bold'))
takeImg.place(x=15, y=500)

trainImg = tk.Button(window, text="Train Image", command=TrainImage, fg="red", bg="yellow", width=20, height=3, activebackground="Red", font=('times', 15, 'bold'))
trainImg.place(x=345, y=500)

trackImg = tk.Button(window, text="Track Images", command=TrackImages, fg="red", bg="yellow", width=20, height=3, activebackground="Red", font=('times', 15, 'bold'))
trackImg.place(x=670, y=500)

quitWindow = tk.Button(window, text="Quit", command=window.destroy, fg="red", bg="yellow", width=17, height=3, activebackground="Red", font=('times', 15, 'bold'))
quitWindow.place(x=990, y=500)



