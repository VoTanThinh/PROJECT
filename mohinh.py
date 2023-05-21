from email.mime import message
import cv2
import tkinter as tk
from tkinter.ttk import *
from tkinter import *
from PIL import ImageTk ,Image
from matplotlib.pyplot import show
import numpy as np
from tkinter import messagebox
from tensorflow import keras
from keras.utils import load_img
from keras.utils.image_utils import img_to_array

#Load Model
model = keras.models.load_model('CUOIKY.h5') 

# Create the GUI window
from tkinter import filedialog
window = tk.Tk()                       # tạo cữa sổ
window.title("FLOWERING DETECTION")    #tạo tiêu đề
window.geometry("800x600")             # kích thước cữa sổ

# show a label
label = Label(window, text='NHẬN DẠNG CÁC LOÀI HOA SỬ DỤNG MÔ HÌNH CNN', font=("Arial Bold", 20),fg ='green')
label.pack(ipadx=100, ipady=10)
label = Label(window, text='GVHD: NGUYỄN TRƯỜNG THỊNH', font=("Arial", 15),fg ='blue')
label.place(x=0,y=60)
label = Label(window, text='SVTH: VÕ TẤN THỊNH', font=("Arial", 15),fg ='blue')
label.place(x=0,y=90)
label = Label(window, text='MSSV: 20146536', font=("Arial", 15),fg ='blue')
label.place(x=0,y=120)

# Tạo label để hiển thị kết quả nhận dạng lên cữa sổ
result_label = tk.Label(window, text="CHƯA CÓ DỮ LIỆU ", font=("Arial", 15),background='red',fg='yellow')
result_label.place(x=145,y=165)

#load ảnh poster
img_poster = (Image.open('poster.jpg'))
resize = img_poster.resize((250,280), Image.ANTIALIAS)
img = ImageTk.PhotoImage(resize)

hinh_anh= Button(window, text='',font=("Arial", 15),image = img)
hinh_anh.place(x=550,y=60)

#load ảnh AI
img_AI = (Image.open('imageAI.jpg'))
resize = img_AI.resize((180,230), Image.ANTIALIAS)
im2 = ImageTk.PhotoImage(resize)

hinhAI= Button(window, text='',font=("Arial", 15),image = im2)
hinhAI.place(x=0,y=380)

# tạo nút nhân thoát giao diện
wd = None
def DongW():
    global wd
    window.destroy()
BT = Button(wd,text="CLOSE",width=10,height=2,font=("Arial", 15),background='green', command=DongW)
BT.place(x=600,y=500)


# Hàm function đê nhận dạng các loài hoa
def predict_disease(image_path):
    img = Image.open(image_path).resize((300, 300))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Đưa ra dự toán từ mô hình đã được training
    prediction = model.predict(img_array)
    # Assuming the model outputs one-hot encoded labels, convert prediction to class labels
    class_labels = ['HOACUC','HOAHONG','HOAHUONGDUONG','HOALY', 'HOAMAI']
    predicted_label = class_labels[np.argmax(prediction)]
    result_label.config(text="KẾT QUẢ NHẬN DẠNG: " + predicted_label)

# Hàm function tạo ra sự kiên nút nhấn
def open_image():
    image_path = filedialog.askopenfilename(initialdir="test_images", title="Select Image",
                                            filetypes=(("JPEG files", "*.jpg"), ("PNG files", "*.png")))
    predict_disease(image_path)
    
    # Hiển thị các hình ảnh đã chọn
    img = Image.open(image_path).resize((300, 300))
    img_tk = ImageTk.PhotoImage(img)
    img_label = tk.Label(window, image=img_tk)
    img_label.image = img_tk
    img_label.place(x=215,y=200)

# Tạo nút button để mở hình ảnh nhận dạng
open_button = tk.Button(window, text="LOAD IMAGE ",font=("Arial", 15),background='orange',fg='white', command=open_image)
open_button.place(x=35,y=270)

window.mainloop()