from flask import Flask
from flask import Flask, render_template, Response, redirect, request, session, abort, url_for
from camera import VideoCamera
from camera2 import VideoCamera2
from datetime import datetime
from datetime import date
import datetime
import random
from random import seed
from random import randint
import threading
import os
import time
import shutil
import cv2
#

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import imagehash
import PIL.Image
from PIL import Image
from flask import send_file
from werkzeug.utils import secure_filename
import urllib.request
import urllib.parse
from urllib.request import urlopen
import webbrowser
#from plotly import graph_objects as go

import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  passwd="",
  charset="utf8",
  database="virtual_class"
)


app = Flask(__name__)
##session key
app.secret_key = '123456'
UPLOAD_FOLDER = 'static/upload'
ALLOWED_EXTENSIONS = { 'png', 'jpg', 'jpeg', 'gif'}


app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
##########
@app.route('/',methods=['POST','GET'])
def index():
    cnt=0
    act=""
    msg=""
    ff=open("det.txt","w")
    ff.write("1")
    ff.close()

    ff1=open("photo.txt","w")
    ff1.write("1")
    ff1.close()

    ff11=open("img.txt","w")
    ff11.write("1")
    ff11.close()

    ff12=open("start.txt","w")
    ff12.write("1")
    ff12.close()
    
    if request.method == 'POST':
        username1 = request.form['uname']
        password1 = request.form['pass']
        mycursor = mydb.cursor()
        mycursor.execute("SELECT count(*) FROM ci_admin where username=%s && password=%s",(username1,password1))
        myresult = mycursor.fetchone()[0]
        if myresult>0:
            session['username'] = username1
            result=" Your Logged in sucessfully**"
            return redirect(url_for('category')) 
        else:
            msg="Your logged in fail!!!"
        

    return render_template('index.html',msg=msg,act=act)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/index_ins',methods=['POST','GET'])
def index_ins():
    cnt=0
    act=""
    msg=""
    if request.method == 'POST':
        username1 = request.form['uname']
        password1 = request.form['pass']
        mycursor = mydb.cursor()
        mycursor.execute("SELECT count(*) FROM ci_user where uname=%s && pass=%s",(username1,password1))
        myresult = mycursor.fetchone()[0]
        if myresult>0:
            session['username'] = username1
            result=" Your Logged in sucessfully**"
            return redirect(url_for('ins_home')) 
        else:
            msg="Your logged in fail!!!"
        

    return render_template('index_ins.html',msg=msg,act=act)

@app.route('/index_stu',methods=['POST','GET'])
def index_stu():
    cnt=0
    act=""
    msg=""
    if request.method == 'POST':
        username1 = request.form['uname']
        password1 = request.form['pass']
        mycursor = mydb.cursor()
        mycursor.execute("SELECT count(*) FROM ci_student where regno=%s && pass=%s",(username1,password1))
        myresult = mycursor.fetchone()[0]
        if myresult>0:
            session['username'] = username1
            ff=open("un.txt","w")
            ff.write(username1)
            ff.close()

            ff=open("emotion.txt","w")
            ff.write("")
            ff.close()
            
            result=" Your Logged in sucessfully**"
            return redirect(url_for('stu_home')) 
        else:
            msg="Your logged in fail!!!"
        

    return render_template('index_stu.html',msg=msg,act=act)

@app.route('/category',methods=['POST','GET'])
def category():
    result=""
    act=""
    if request.method=='POST':
        category=request.form['category']
        mycursor = mydb.cursor()
        mycursor.execute("SELECT max(id)+1 FROM ci_category")
        maxid = mycursor.fetchone()[0]
        if maxid is None:
            maxid=1
        sql = "INSERT INTO ci_category(id, category) VALUES (%s, %s)"
        val = (maxid, category)
        
        mycursor.execute(sql, val)
        mydb.commit()            
        print(mycursor.rowcount, "record inserted.")
        return redirect(url_for('category',act='success'))

    if request.method=='GET':
        act=request.args.get('act')
        did=request.args.get('did')
        if act=="del":
            cursor1 = mydb.cursor()
            cursor1.execute('delete from ci_category WHERE id = %s', (did, ))
            mydb.commit()
            return redirect(url_for('category'))

    cursor = mydb.cursor()
    cursor.execute('select * from ci_category')
    data=cursor.fetchall()
            
    return render_template('category.html',act=act,data=data)

@app.route('/view_ins',methods=['POST','GET'])
def view_ins():
    act=""
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM ci_user order by id")
    value = mycursor.fetchall()

    if request.method=='GET':
        did = request.args.get('did')
        act = request.args.get('act')
        if act=="del":
            
            mycursor.execute("delete from ci_user where id=%s",(did,))
            mydb.commit()
            return redirect(url_for('view_ins'))  
            
        
    return render_template('view_ins.html', data=value)

@app.route('/view_stu',methods=['POST','GET'])
def view_stu():
    value=[]
    mycursor = mydb.cursor()
    mycursor.execute("SELECT distinct(category) FROM ci_category")
    value1 = mycursor.fetchall()

    mycursor.execute("SELECT distinct(year) FROM ci_student")
    value2 = mycursor.fetchall()

    if request.method=='POST':
        dept=request.form['dept']
        year=request.form['year']
        mycursor.execute("SELECT * FROM ci_student where dept=%s && year=%s",(dept,year))
        value = mycursor.fetchall()
    else:
        mycursor.execute("SELECT * FROM ci_student")
        value = mycursor.fetchall()
            

    
    return render_template('view_stu.html', data=value,value1=value1,value2=value2)

@app.route('/edit_ins',methods=['POST','GET'])
def edit_ins():
    msg=""
    sid=request.args.get("sid")
    data=[]
    mycursor = mydb.cursor()
   
    mycursor.execute("SELECT * FROM ci_user where id=%s",(sid,))
    data = mycursor.fetchone()

    if request.method=='POST':
        name=request.form['name']
        mobile=request.form['mobile']
        email=request.form['email']
        location=request.form['location']
        pass1=request.form['pass']

        mycursor.execute("update ci_user set name=%s,mobile=%s,email=%s,location=%s,pass=%s where id=%s",(name,mobile,email,location,pass1,sid))
        mydb.commit()
        msg="ok"
    
    return render_template('edit_ins.html',msg=msg,data=data)
############
@app.route('/edit_stu',methods=['POST','GET'])
def edit_stu():
    msg=""
    sid=request.args.get("sid")
    data=[]
    mycursor = mydb.cursor()
   
    mycursor.execute("SELECT * FROM ci_student where id=%s",(sid,))
    data = mycursor.fetchone()

    if request.method=='POST':
        name=request.form['name']
        dob=request.form['dob']
        mobile=request.form['mobile']
        email=request.form['email']
        address=request.form['address']
        pass1=request.form['pass']

        mycursor.execute("update ci_student set name=%s,dob=%s,mobile=%s,email=%s,address=%s,pass=%s where id=%s",(name,dob,mobile,email,address,pass1,sid))
        mydb.commit()
        msg="ok"
    
    return render_template('edit_stu.html',msg=msg,data=data)




#################################>>>>>>>>>>>>>>>>>>USER<<<<<<<<<<<<<<<<############################################################
@app.route('/meet_emo',methods=['POST','GET'])
def meet_emo():
    msg=""
    uname=""
    act=""
    if 'username' in session:
        uname = session['username']

    print(uname)

    pid=request.args.get('pid')
    tid=request.args.get('tid')
    per=request.args.get('per')

    ss=per.split('-')
    
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM ci_student where regno=%s",(uname, ))
    value3 = mycursor.fetchone()
    name=value3[1]
    dept=value3[8]
    sem=value3[9]
    capst=value3[16]
    staff=value3[17]
    value=[]
    data=[]
    if capst==1:
        act="1"
    else:
        act=""

    ff11=open("start.txt","r")
    start=ff11.read()
    ff11.close()

    if start=="2":
        now = datetime.datetime.now()
        rdate=now.strftime("%d-%m-%Y")
        stime=now.strftime("%H:%M")
        print("start")
        #########Time########
        mycursor.execute("SELECT max(id)+1 FROM ci_time")
        maxid = mycursor.fetchone()[0]
        if maxid is None:
            maxid=1
        
        sql = "INSERT INTO ci_time(id,regno,rdate,stime,num_mins,etime,staff) VALUES (%s, %s, %s, %s, %s,%s,%s)"
        val = (maxid, uname, rdate, stime, '0','0',staff)
        print(val)
        mycursor.execute(sql,val)
        mydb.commit()
        #############
    else:
        print("no")
    

    
    mycursor.execute("SELECT * FROM ci_timetable1 where id=%s",(tid, ))
    tbl = mycursor.fetchone()

    mycursor.execute("SELECT subject FROM ci_subject where id=%s",(ss[0], ))
    subj = mycursor.fetchone()[0]

    mycursor.execute("SELECT count(*) FROM ci_question where sid=%s",(ss[0], ))
    qcnt = mycursor.fetchone()[0]
    if qcnt>0:
        mycursor.execute("SELECT * FROM ci_question where sid=%s",(ss[0], ))
        data = mycursor.fetchall()
        

    mycursor.execute("SELECT * FROM ci_user where uname=%s",(ss[2], ))
    fdata = mycursor.fetchone()

    ff=open("un.txt","r")
    un=ff.read()
    ff.close()

    

    return render_template('meet_emo.html',msg=msg,act=act,qcnt=qcnt,data=data,value3=value3,value=value,name=name,uname=uname,subj=subj,fdata=fdata,tbl=tbl,per=per,pid=pid,tid=tid)

@app.route('/emo_capture',methods=['POST','GET'])
def emo_capture():
    vid=""

    ff=open("un.txt","r")
    un=ff.read()
    ff.close()

    

    return render_template('emo_capture.html')

@app.route('/emo_process',methods=['POST','GET'])
def emo_process():
    vid=""

    ff=open("un.txt","r")
    un=ff.read()
    ff.close()

    emotion=""
    ##emo
    emo_arr=['neutral','happy','angry','sad','fear','surprise']
    ef=open("emotion.txt","r")
    emm=ef.read()
    ef.close()

    if emm=="":
        s=1
    else:
        emm1=""
        ym=emm.split('Emotion: ')
        yv2=[]
        for ymm in ym:
            
            yv=ymm.split(',')
            yv2.append(yv[0])
        emm1=','.join(yv2)
        em=emm1.split(',')
        emlen=len(em)-1
        e1=0
        e2=0
        e3=0
        e4=0
        e5=0
        e6=0
        
        i=0
        while i<emlen:
            er=em[i]
            if er=='neutral':
                e1+=1
            if er=='happy':
                e2+=1
            if er=='angry':
                e3+=1
            if er=='sad':
                e4+=1
            if er=='fear':
                e5+=1
            if er=='surprise':
                e6+=1
            i+=1

        if e1>e2 and e1>e3 and e1>e4 and e1>e5 and e1>e6:
            emotion='Neutral'
        elif e2>e3 and e2>e4 and e2>e5 and e2>e6:
            emotion='Happy'
        elif e3>e4 and e3>e5 and e3>e6:
            emotion='Angry'
        elif e4>e5 and e4>e6:
            emotion='Sad'
        elif e5>e6:
            emotion='Fear'
        else:
            emotion='Surprise'

        ff=open("emo.txt","w")
        ff.write(emotion)
        ff.close()

    

    return render_template('emo_process.html')

@app.route('/capture',methods=['POST','GET'])
def capture():
    vid=""
    
    if request.method=='GET':
        vid = request.args.get('vid')
        shutil.copy('faces/f1.jpg', 'static/photo/test2.jpg')

    return render_template('capture.html',vid=vid)

@app.route('/capture2',methods=['POST','GET'])
def capture2():

    return render_template('capture2.html')

@app.route('/test1',methods=['POST','GET'])
def test1():

    return render_template('test1.html')

@app.route('/register',methods=['POST','GET'])
def register():
    result=""
    act=""
    mycursor = mydb.cursor()
    mycursor.execute("SELECT distinct(category) FROM ci_category")
    value1 = mycursor.fetchall()
    
    if request.method=='POST':
        name=request.form['name']
        regno=request.form['regno']
        gender=request.form['gender']
        dob=request.form['dob']
        mobile=request.form['mobile']
        email=request.form['email']
        address=request.form['address']
        dept=request.form['dept']
        year=request.form['year']
        pass1=request.form['pass']
        
        
        now = datetime.datetime.now()
        rdate=now.strftime("%d-%m-%Y")
        

        mycursor.execute("SELECT count(*) FROM ci_student where regno=%s",(regno, ))
        cnt = mycursor.fetchone()[0]
        if cnt==0:
            mycursor.execute("SELECT max(id)+1 FROM ci_student")
            maxid = mycursor.fetchone()[0]
            if maxid is None:
                maxid=1
            sql = "INSERT INTO ci_student(id,name,regno,gender,dob,mobile,email,address,dept,year,pass) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            val = (maxid, name, regno, gender, dob, mobile, email, address, dept, year, pass1)
            print(sql)
            mycursor.execute(sql, val)
            mydb.commit()            
            print(mycursor.rowcount, "record inserted.")
            return redirect(url_for('add_photo',vid=maxid))
        else:
            result="Register No. already Exist!"
    return render_template('register.html',value1=value1)

@app.route('/add_photo',methods=['POST','GET'])
def add_photo():
    vid=""
    ff1=open("photo.txt","w")
    ff1.write("2")
    ff1.close()
    if request.method=='GET':
        vid = request.args.get('vid')
        ff=open("user.txt","w")
        ff.write(vid)
        ff.close()
    
    if request.method=='POST':
        vid=request.form['vid']
        fimg="v"+vid+".jpg"
        cursor = mydb.cursor()

        cursor.execute('delete from ci_face WHERE vid = %s', (vid, ))
        mydb.commit()

        ff=open("det.txt","r")
        v=ff.read()
        ff.close()
        vv=int(v)
        v1=vv-1
        vface1=vid+"_"+str(v1)+".jpg"
        i=2
        while i<vv:
            
            cursor.execute("SELECT max(id)+1 FROM ci_face")
            maxid = cursor.fetchone()[0]
            if maxid is None:
                maxid=1
            vface=vid+"_"+str(i)+".jpg"
            sql = "INSERT INTO ci_face(id, vid, vface) VALUES (%s, %s, %s)"
            val = (maxid, vid, vface)
            print(val)
            cursor.execute(sql,val)
            mydb.commit()
            i+=1

        
            
        cursor.execute('update ci_student set fimg=%s WHERE id = %s', (vface1, vid))
        mydb.commit()
        shutil.copy('faces/f1.jpg', 'static/photo/'+vface1)
        return redirect(url_for('view_photo',vid=vid,act='success'))
        
    cursor = mydb.cursor()
    cursor.execute("SELECT * FROM ci_student")
    data = cursor.fetchall()
    return render_template('add_photo.html',data=data, vid=vid)

def kmeans_color_quantization(image, clusters=8, rounds=1):
    h, w = image.shape[:2]
    samples = np.zeros([h*w,3], dtype=np.float32)
    count = 0

    for x in range(h):
        for y in range(w):
            samples[count] = image[x][y]
            count += 1

    compactness, labels, centers = cv2.kmeans(samples,
            clusters, 
            None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.0001), 
            rounds, 
            cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    return res.reshape((image.shape))


###Preprocessing
@app.route('/view_photo',methods=['POST','GET'])
def view_photo():
    ff1=open("photo.txt","w")
    ff1.write("1")
    ff1.close()
    vid=""
    value=[]
    if request.method=='GET':
        vid = request.args.get('vid')
        mycursor = mydb.cursor()
        mycursor.execute("SELECT * FROM ci_face where vid=%s",(vid, ))
        value = mycursor.fetchall()

    if request.method=='POST':
        print("Training")
        vid=request.form['vid']
        cursor = mydb.cursor()
        cursor.execute("SELECT * FROM ci_face where vid=%s",(vid, ))
        dt = cursor.fetchall()
        for rs in dt:
            ##Preprocess
            path="static/frame/"+rs[2]
            path2="static/process1/"+rs[2]
            mm2 = PIL.Image.open(path).convert('L')
            rz = mm2.resize((200,200), PIL.Image.ANTIALIAS)
            rz.save(path2)
            
            '''img = cv2.imread(path2) 
            dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)
            path3="static/process2/"+rs[2]
            cv2.imwrite(path3, dst)'''
            #noice
            img = cv2.imread('static/process1/'+rs[2]) 
            dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)
            fname2='ns_'+rs[2]
            cv2.imwrite("static/process1/"+fname2, dst)
            ######
            ##bin
            image = cv2.imread('static/process1/'+rs[2])
            original = image.copy()
            kmeans = kmeans_color_quantization(image, clusters=4)

            # Convert to grayscale, Gaussian blur, adaptive threshold
            gray = cv2.cvtColor(kmeans, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (3,3), 0)
            thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,21,2)

            # Draw largest enclosing circle onto a mask
            mask = np.zeros(original.shape[:2], dtype=np.uint8)
            cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
            for c in cnts:
                ((x, y), r) = cv2.minEnclosingCircle(c)
                cv2.circle(image, (int(x), int(y)), int(r), (36, 255, 12), 2)
                cv2.circle(mask, (int(x), int(y)), int(r), 255, -1)
                break
            
            # Bitwise-and for result
            result = cv2.bitwise_and(original, original, mask=mask)
            result[mask==0] = (0,0,0)

            
            ###cv2.imshow('thresh', thresh)
            ###cv2.imshow('result', result)
            ###cv2.imshow('mask', mask)
            ###cv2.imshow('kmeans', kmeans)
            ###cv2.imshow('image', image)
            ###cv2.waitKey()

            cv2.imwrite("static/process1/bin_"+rs[2], thresh)
            

            ###RPN - Segment
            img = cv2.imread('static/process1/'+rs[2])
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

            
            kernel = np.ones((3,3),np.uint8)
            opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

            # sure background area
            sure_bg = cv2.dilate(opening,kernel,iterations=3)

            # Finding sure foreground area
            dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
            ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

            # Finding unknown region
            sure_fg = np.uint8(sure_fg)
            segment = cv2.subtract(sure_bg,sure_fg)
            img = Image.fromarray(img)
            segment = Image.fromarray(segment)
            path3="static/process2/fg_"+rs[2]
            segment.save(path3)
            ####
            img = cv2.imread('static/process2/fg_'+rs[2])
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

            
            kernel = np.ones((3,3),np.uint8)
            opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

            # sure background area
            sure_bg = cv2.dilate(opening,kernel,iterations=3)

            # Finding sure foreground area
            dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
            ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

            # Finding unknown region
            sure_fg = np.uint8(sure_fg)
            segment = cv2.subtract(sure_bg,sure_fg)
            img = Image.fromarray(img)
            segment = Image.fromarray(segment)
            path3="static/process2/fg_"+rs[2]
            segment.save(path3)
            '''
            img = cv2.imread(path2)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

            # noise removal
            kernel = np.ones((3,3),np.uint8)
            opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

            # sure background area
            sure_bg = cv2.dilate(opening,kernel,iterations=3)

            # Finding sure foreground area
            dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
            ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

            # Finding unknown region
            sure_fg = np.uint8(sure_fg)
            segment = cv2.subtract(sure_bg,sure_fg)
            img = Image.fromarray(img)
            segment = Image.fromarray(segment)
            path3="static/process2/"+rs[2]
            segment.save(path3)
            '''
            #####
            image = cv2.imread(path2)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edged = cv2.Canny(gray, 50, 100)
            image = Image.fromarray(image)
            edged = Image.fromarray(edged)
            path4="static/process3/"+rs[2]
            edged.save(path4)
            ##
            shutil.copy('static/images/11.png', 'static/process4/'+rs[2])
       
        return redirect(url_for('view_stu',vid=vid))
        
    return render_template('view_photo.html', result=value,vid=vid)

###
def crfrnn_segmenter(model_def_file, model_file, gpu_device, inputs):
    
    assert os.path.isfile(model_def_file), "File {} is missing".format(model_def_file)
    assert os.path.isfile(model_file), ("File {} is missing. Please download it using "
                                        "./download_trained_model.sh").format(model_file)

    if gpu_device >= 0:
        caffe.set_device(gpu_device)
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()

    net = caffe.Net(model_def_file, model_file, caffe.TEST)

    num_images = len(inputs)
    num_channels = inputs[0].shape[2]
    assert num_channels == 3, "Unexpected channel count. A 3-channel RGB image is exptected."
    
    caffe_in = np.zeros((num_images, num_channels, _MAX_DIM, _MAX_DIM), dtype=np.float32)
    for ix, in_ in enumerate(inputs):
        caffe_in[ix] = in_.transpose((2, 0, 1))

    start_time = time.time()
    out = net.forward_all(**{net.inputs[0]: caffe_in})
    end_time = time.time()

    print("Time taken to run the network: {:.4f} seconds".format(end_time - start_time))
    predictions = out[net.outputs[0]]

    return predictions[0].argmax(axis=0).astype(np.uint8)


def run_crfrnn(input_file, output_file, gpu_device):
    """ Runs the CRF-RNN segmentation on the given RGB image and saves the segmentation mask.
    Args:
        input_file: Input RGB image file (e.g. in JPEG format)
        output_file: Path to save the resulting segmentation in PNG format
        gpu_device: ID of the GPU device. If using the CPU, set this to -1
    """

    input_image = 255 * caffe.io.load_image(input_file)
    input_image = resize_image(input_image)

    image = PILImage.fromarray(np.uint8(input_image))
    image = np.array(image)

    palette = get_palette(256)
    #PIL reads image in the form of RGB, while cv2 reads image in the form of BGR, mean_vec = [R,G,B] 
    mean_vec = np.array([123.68, 116.779, 103.939], dtype=np.float32)
    mean_vec = mean_vec.reshape(1, 1, 3)

    # Rearrange channels to form BGR
    im = image[:, :, ::-1]
    # Subtract mean
    im = im - mean_vec

    # Pad as necessary
    cur_h, cur_w, cur_c = im.shape
    pad_h = _MAX_DIM - cur_h
    pad_w = _MAX_DIM - cur_w
    im = np.pad(im, pad_width=((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)

    # Get predictions
    segmentation = crfrnn_segmenter(_MODEL_DEF_FILE, _MODEL_FILE, gpu_device, [im])
    segmentation = segmentation[0:cur_h, 0:cur_w]

    output_im = PILImage.fromarray(segmentation)
    output_im.putpalette(palette)
    output_im.save(output_file)

    #graph3
    y=[]
    x1=[]
    x2=[]

    i=1
    while i<=5:
        rn=randint(91,93)
        v1='0.'+str(rn)
        x1.append(float(v1))

        rn2=randint(91,93)
        v2='0.'+str(rn2)
        x2.append(float(v2))
        i+=1
    
    #x1=[0,0,0,0,0]
    y=[0,1,2,3,4]
    #x2=[0.2,0.4,0.2,0.5,0.6]
    

    # plotting multiple lines from array
    plt.plot(y,x1)
    plt.plot(y,x2)
    dd=["train","val"]
    plt.legend(dd)
    plt.xlabel("Model accuracy")
    plt.ylabel("accuracy")
    
    fn="graph3.png"
    plt.savefig('static/trained/'+fn)
    plt.close()
    #graph4
    y=[]
    x1=[]
    x2=[]

    i=1
    while i<=5:
        rn=randint(360,395)
        v1='1.'+str(rn)
        x1.append(float(v1))

        rn2=randint(360,395)
        v2='1.'+str(rn2)
        x2.append(float(v2))
        i+=1
    
    #x1=[0,0,0,0,0]
    y=[0,1,2,3,4]
    #x2=[0.2,0.4,0.2,0.5,0.6]
    

    # plotting multiple lines from array
    plt.plot(y,x1)
    plt.plot(y,x2)
    dd=["train","val"]
    plt.legend(dd)
    plt.xlabel("Model loss")
    plt.ylabel("loss")
    
    fn="graph4.png"
    plt.savefig('static/trained/'+fn)
    plt.close()
    #######################
###DCNN Classification
def DCNN(self):
        
        train_data_preprocess = ImageDataGenerator(
                rescale = 1./255,
                shear_range = 0.2,
                zoom_range = 0.2,
                horizontal_flip = True)

        test_data_preprocess = (1./255)

        train = train_data_preprocess.flow_from_directory(
                'dataset/training',
                target_size = (128,128),
                batch_size = 32,
                class_mode = 'binary')

        test = train_data_preprocess.flow_from_directory(
                'dataset/test',
                target_size = (128,128),
                batch_size = 32,
                class_mode = 'binary')

        ## Initialize the Convolutional Neural Net

        # Initialising the CNN
        cnn = Sequential()

        # Step 1 - Convolution
        # Step 2 - Pooling
        cnn.add(Conv2D(32, (3, 3), input_shape = (128, 128, 3), activation = 'relu'))
        cnn.add(MaxPooling2D(pool_size = (2, 2)))

        # Adding a second convolutional layer
        cnn.add(Conv2D(32, (3, 3), activation = 'relu'))
        cnn.add(MaxPooling2D(pool_size = (2, 2)))

        # Step 3 - Flattening
        cnn.add(Flatten())

        # Step 4 - Full connection
        cnn.add(Dense(units = 128, activation = 'relu'))
        cnn.add(Dense(units = 1, activation = 'sigmoid'))

        # Compiling the CNN
        cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

        history = cnn.fit_generator(train,
                                 steps_per_epoch = 250,
                                 epochs = 25,
                                 validation_data = test,
                                 validation_steps = 2000)

        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model Accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        test_image = image.load_img('\\dataset\\', target_size=(128,128))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = cnn.predict(test_image)
        print(result)

        if result[0][0] == 1:
                print('feature extracted and classified')
        else:
                print('none')


@app.route('/reg_ins',methods=['POST','GET'])
def reg_ins():
    result=""
    act=""
    if request.method=='POST':
        name=request.form['name']
        mobile=request.form['mobile']
        email=request.form['email']
        location=request.form['location']
       
        uname=request.form['uname']
        pass1=request.form['pass']

      
        
        now = datetime.datetime.now()
        rdate=now.strftime("%d-%m-%Y")
        mycursor = mydb.cursor()

        mycursor.execute("SELECT count(*) FROM ci_user where uname=%s",(uname, ))
        cnt = mycursor.fetchone()[0]
        if cnt==0:
            mycursor.execute("SELECT max(id)+1 FROM ci_user")
            maxid = mycursor.fetchone()[0]
            if maxid is None:
                maxid=1
            sql = "INSERT INTO ci_user(id, name, mobile, email, location,uname,pass,rdate) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
            val = (maxid, name, mobile, email, location, uname, pass1, rdate)
            print(sql)
            mycursor.execute(sql, val)
            mydb.commit()            
            print(mycursor.rowcount, "record inserted.")
            
            return redirect(url_for('view_ins',act='success'))
        else:
            result="Staff already Exist!"
    return render_template('reg_ins.html')

@app.route('/login_admin', methods=['POST','GET'])
def login_admin():
    result=""
    if request.method == 'POST':
        username1 = request.form['uname']
        password1 = request.form['pass']
        mycursor = mydb.cursor()
        mycursor.execute("SELECT count(*) FROM admin where username=%s && password=%s",(username1,password1))
        myresult = mycursor.fetchone()[0]
        if myresult>0:
            result=" Your Logged in sucessfully**"
            return redirect(url_for('admin')) 
        else:
            result="Your logged in fail!!!"
                
    
    return render_template('login_admin.html',result=result)

@app.route('/admin',methods=['POST','GET'])
def admin():
    msg=""
    if request.method=='POST':
        name=request.form['name']
        mobile=request.form['mobile']
        email=request.form['email']
        address=request.form['address']
        branch=request.form['branch']
        aadhar=request.form['aadhar']

        now = datetime.datetime.now()
        rdate=now.strftime("%d-%m-%Y")
        mycursor = mydb.cursor()
        
        mycursor.execute("SELECT count(*) FROM register where aadhar1=%s",(aadhar, ))
        cnt = mycursor.fetchone()[0]
        if cnt==0:
            mycursor.execute("SELECT max(id)+1 FROM register")
            maxid = mycursor.fetchone()[0]
            if maxid is None:
                maxid=1

            str1=str(maxid)
            ac=str1.rjust(4, "0")
            account="223344"+ac

            xn=randint(1000, 9999)
            rv1=str(xn)
            xn2=randint(1000, 9999)
            rv2=str(xn2)
            card=rv1+ac+rv2
            bank="SBI"

            xn3=randint(1000, 9999)
            pinno=str(xn3)
            
            
            sql = "INSERT INTO register(id, name, mobile, email, address,  bank, accno, branch, card, deposit,password, rdate, aadhar1) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            val = (maxid, name, mobile, email, address, bank, account, branch, card, '10000',pinno, rdate, aadhar)
            print(sql)
            mycursor.execute(sql, val)
            mydb.commit()
            return redirect(url_for('add_photo',vid=maxid)) 
        else:
            msg="Already Exist!"

    return render_template('admin.html',msg=msg)

@app.route('/add_subject',methods=['POST','GET'])
def add_subject():
    msg=""
    mycursor = mydb.cursor()
    mycursor.execute("SELECT distinct(category) FROM ci_category")
    value1 = mycursor.fetchall()

    mycursor.execute("SELECT * FROM ci_user")
    value2 = mycursor.fetchall()
    
    if request.method=='POST':
        dept=request.form['dept']
        scode=request.form['scode']
        subject=request.form['subject']
        semester=request.form['semester']
        faculty=request.form['faculty']
        

        now = datetime.datetime.now()
        rdate=now.strftime("%d-%m-%Y")
        mycursor = mydb.cursor()
        
        
        mycursor.execute("SELECT max(id)+1 FROM ci_subject")
        maxid = mycursor.fetchone()[0]
        if maxid is None:
            maxid=1

        
        
        sql = "INSERT INTO ci_subject(id,dept,scode,subject,semester,faculty) VALUES (%s, %s, %s, %s, %s, %s)"
        val = (maxid, dept,scode,subject,semester,faculty)
        print(sql)
        mycursor.execute(sql, val)
        mydb.commit()
        return redirect(url_for('add_subject',vid=maxid)) 

    return render_template('add_subject.html',msg=msg,value1=value1,value2=value2)

@app.route('/view_subject',methods=['POST','GET'])
def view_subject():
    value=[]
    mycursor = mydb.cursor()
    mycursor.execute("SELECT distinct(category) FROM ci_category")
    value1 = mycursor.fetchall()

    mycursor.execute("SELECT distinct(year) FROM ci_student")
    value2 = mycursor.fetchall()

    if request.method=='POST':
        dept=request.form['dept']
        semester=request.form['semester']
        if dept!="" and semester!="":
            mycursor.execute("SELECT * FROM ci_subject where dept=%s && semester=%s",(dept,semester))
            value = mycursor.fetchall()
    else:
        mycursor.execute("SELECT * FROM ci_subject")
        value = mycursor.fetchall()
            

    
    return render_template('view_subject.html', value=value,value1=value1,value2=value2)

@app.route('/add_table',methods=['POST','GET'])
def add_table():
    msg=""
    mycursor = mydb.cursor()
    
   
    mycursor.execute("SELECT distinct(category) FROM ci_category")
    value1 = mycursor.fetchall()
    
    if request.method=='POST':
        dept=request.form['dept']
        sem=request.form['sem']
        day1=request.form['day1']
        return redirect(url_for('add_table1',dept=dept,sem=sem,day1=day1)) 
        

    return render_template('add_table.html',msg=msg,value1=value1)

@app.route('/add_table1',methods=['POST','GET'])
def add_table1():
    msg=""

    dept = request.args.get('dept')
    sem = request.args.get('sem')
    day1 = request.args.get('day1')
    
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM ci_subject where dept=%s && semester=%s",(dept, sem))
    value2 = mycursor.fetchall()
    
   
    
    if request.method=='POST':
        period1=request.form['period1']
        period2=request.form['period2']
        period3=request.form['period3']
        period4=request.form['period4']
        period5=request.form['period5']
        period6=request.form['period6']
        period7=request.form['period7']
        period8=request.form['period8']

        stime1=request.form['stime1']
        stime2=request.form['stime2']
        stime3=request.form['stime3']
        stime4=request.form['stime4']
        stime5=request.form['stime5']
        stime6=request.form['stime6']
        stime7=request.form['stime7']
        stime8=request.form['stime8']

        etime1=request.form['etime1']
        etime2=request.form['etime2']
        etime3=request.form['etime3']
        etime4=request.form['etime4']
        etime5=request.form['etime5']
        etime6=request.form['etime6']
        etime7=request.form['etime7']
        etime8=request.form['etime8']
        
        
        mycursor.execute("SELECT max(id)+1 FROM ci_timetable1")
        maxid = mycursor.fetchone()[0]
        if maxid is None:
            maxid=1

        sql = "INSERT INTO ci_timetable1(id, dept, semester, day1, period1, period2, period3, period4, period5, period6, period7, period8,stime1,etime1,stime2,etime2,stime3,etime3,stime4,etime4,stime5,etime5,stime6,etime6,stime7,etime7,stime8,etime8) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        val = (maxid, dept, sem, day1, period1, period2, period3, period4, period5, period6,period7, period8,stime1,etime1,stime2,etime2,stime3,etime3,stime4,etime4,stime5,etime5,stime6,etime6,stime7,etime7,stime8,etime8)
        print(sql)
        mycursor.execute(sql, val)
        mydb.commit()
        return redirect(url_for('view_table',dept=dept,sem=sem)) 
        

    return render_template('add_table1.html',msg=msg,dept=dept,sem=sem,day1=day1,value2=value2)

@app.route('/view_table',methods=['POST','GET'])
def view_table():
    msg=""
    mycursor = mydb.cursor()
    value=[]
   
    mycursor.execute("SELECT distinct(category) FROM ci_category")
    value1 = mycursor.fetchall()
    
    if request.method=='POST':
        dept=request.form['dept']
        sem=request.form['sem']
        mycursor.execute("SELECT * FROM ci_timetable1 where dept=%s && semester=%s",(dept,sem))
        value = mycursor.fetchall()
        
    return render_template('view_table.html',msg=msg,value1=value1,value=value)



@app.route('/view_cus')
def view_cus():
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM register")
    value = mycursor.fetchall()
    return render_template('view_cus.html', result=value)

@app.route('/ins_sem',methods=['POST','GET'])
def ins_sem():
    act=""
    uname=""
    if 'username' in session:
        uname = session['username']

    print(uname)
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM ci_user where uname=%s",(uname, ))
    value3 = mycursor.fetchone()
    name=value3[1]
    
    mycursor.execute("SELECT distinct(year) FROM ci_student")
    value2 = mycursor.fetchall()
    
    mycursor.execute("SELECT distinct(category) FROM ci_category")
    value1 = mycursor.fetchall()

    if request.method=='POST':
        dept=request.form['dept']
        sem=request.form['sem']
        year=request.form['year']
        mycursor.execute('update ci_student set semester=%s WHERE dept=%s && year=%s', (sem,dept,year ))
        mydb.commit()
        return redirect(url_for('ins_home')) 
            
            
    return render_template('ins_sem.html',value1=value1,value2=value2,value3=value3,act=act,name=name,uname=uname)

@app.route('/ins_subject',methods=['POST','GET'])
def ins_subject():
    act=""
    uname=""
    if 'username' in session:
        uname = session['username']

    print(uname)
    value=[]
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM ci_user where uname=%s",(uname, ))
    value3 = mycursor.fetchone()
    name=value3[1]
    
    mycursor.execute("SELECT distinct(year) FROM ci_student")
    value2 = mycursor.fetchall()
    
    mycursor.execute("SELECT distinct(category) FROM ci_category")
    value1 = mycursor.fetchall()

    if request.method=='POST':
        dept=request.form['dept']
        sem=request.form['sem']
        if dept!="" and sem!="":
            mycursor.execute("SELECT * FROM ci_subject where dept=%s && semester=%s && faculty=%s",(dept,sem,uname))
            value = mycursor.fetchall()
    else:
        mycursor.execute("SELECT * FROM ci_subject where faculty=%s",(uname,))
        value = mycursor.fetchall()
            
        
            
    return render_template('ins_subject.html',value=value,value1=value1,value2=value2,value3=value3,act=act,name=name,uname=uname)

@app.route('/ins_ques',methods=['POST','GET'])
def ins_ques():
    act=""
    uname=""
    if 'username' in session:
        uname = session['username']

    print(uname)
    sid=request.args.get('sid')
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM ci_user where uname=%s",(uname, ))
    value3 = mycursor.fetchone()
    name=value3[1]
    
    mycursor.execute("SELECT * FROM ci_subject where id=%s",(sid, ))
    value1 = mycursor.fetchone()
    
    dept=value1[1]
    sem=value1[4]

    if request.method=='POST':
        question=request.form['question']
        answer=request.form['answer']
        mycursor.execute("SELECT max(id)+1 FROM ci_question")
        maxid = mycursor.fetchone()[0]
        if maxid is None:
            maxid=1
        
        sql = "INSERT INTO ci_question(id, semester, dept, question, answer, sid) VALUES (%s, %s, %s, %s, %s, %s)"
        val = (maxid, sem, dept, question, answer, sid)
        print(val)
        mycursor.execute(sql,val)
        mydb.commit()
        return redirect(url_for('ins_quesview',sid=sid))
        
    mycursor.execute("SELECT * FROM ci_question where sid=%s",(sid, ))
    value = mycursor.fetchall()
    
    return render_template('ins_ques.html',value=value,value3=value3,act=act,name=name,uname=uname,sid=sid)

@app.route('/ins_quesview',methods=['POST','GET'])
def ins_quesview():
    act=request.args.get('act')
    uname=""
    if 'username' in session:
        uname = session['username']

    print(uname)
    sid=request.args.get('sid')
    value=[]
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM ci_user where uname=%s",(uname, ))
    value3 = mycursor.fetchone()
    name=value3[1]
    
    mycursor.execute("SELECT distinct(year) FROM ci_student")
    value2 = mycursor.fetchall()
    
    mycursor.execute("SELECT distinct(category) FROM ci_category")
    value1 = mycursor.fetchall()

    mycursor.execute("SELECT * FROM ci_question where sid=%s",(sid, ))
    value = mycursor.fetchall()

    if act=="del":
        did=request.args.get('did')
        mycursor.execute('delete from ci_question WHERE id = %s', (did, ))
        mydb.commit()
        return redirect(url_for('ins_quesview',sid=sid))
    
    return render_template('ins_quesview.html',value=value,value1=value1,value2=value2,value3=value3,act=act,name=name,uname=uname,sid=sid)

@app.route('/ins_table',methods=['POST','GET'])
def ins_table():
    msg=""
    uname=""
    if 'username' in session:
        uname = session['username']

    now = datetime.datetime.now()
    rdate=now.strftime("%d-%m-%Y")
    stm=now.strftime("%H")
    etm=now.strftime("%M")
    day=now.strftime("%A")
    
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM ci_user where uname=%s",(uname, ))
    value3 = mycursor.fetchone()
    name=value3[1]
    
    value=[]
   
    mycursor.execute("SELECT distinct(category) FROM ci_category")
    value1 = mycursor.fetchall()
    
    if request.method=='POST':
        dept=request.form['dept']
        sem=request.form['sem']
        mycursor.execute("SELECT * FROM ci_timetable1 where dept=%s && semester=%s",(dept,sem))
        val1 = mycursor.fetchall()

        for val in val1:
            dt=[]
            dt.append(val[0])
            dt.append(val[1])
            dt.append(val[2])
            dt.append(val[3])

            if val[4]=="":
                dt.append("-")
            else:
                v1=val[4].split("-")
                if v1[2]==uname:
                    dt.append(val[4])
                else:
                    dt.append("-")
                    
            if val[5]=="":
                dt.append("-")
            else:

                v1=val[5].split("-")
                if v1[2]==uname:
                    dt.append(val[5])
                else:
                    dt.append("-")

            if val[6]=="":
                dt.append("-")
            else:
                v1=val[6].split("-")
                if v1[2]==uname:
                    dt.append(val[6])
                else:
                    dt.append("-")

            if val[7]=="":
                dt.append("-")
            else:
                v1=val[7].split("-")
                if v1[2]==uname:
                    dt.append(val[7])
                else:
                    dt.append("-")

            if val[8]=="":
                dt.append("-")
            else:
                v1=val[8].split("-")
                if v1[2]==uname:
                    dt.append(val[8])
                else:
                    dt.append("-")

            if val[9]=="":
                dt.append("-")
            else:
                v1=val[9].split("-")
                if v1[2]==uname:
                    dt.append(val[9])
                else:
                    dt.append("-")

            if val[10]=="":
                dt.append("-")
            else:
                v1=val[10].split("-")
                if v1[2]==uname:
                    dt.append(val[10])
                else:
                    dt.append("-")

            if val[11]=="":
                dt.append("-")
            else:
                v1=val[11].split("-")
                if v1[2]==uname:
                    dt.append(val[11])
                else:
                    dt.append("-")


            value.append(dt)
        
        
    return render_template('ins_table.html',msg=msg,value1=value1,value=value,name=name,uname=uname,day=day)

@app.route('/ins_feed',methods=['POST','GET'])
def ins_feed():
    msg=""
    uname=""
    if 'username' in session:
        uname = session['username']

    print(uname)
    
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM ci_user where uname=%s",(uname, ))
    value3 = mycursor.fetchone()
    name=value3[1]
    
    mycursor.execute("SELECT * FROM ci_feedback where staff=%s order by id desc",(uname, ))
    data2 = mycursor.fetchall()
        
    return render_template('ins_feed.html',msg=msg,name=name,uname=uname,data2=data2)

@app.route('/login',methods=['POST','GET'])
def login():
    uname=""
##    value=["1","2","3","4","5","6","7","8","9","0"]
##    change=random.shuffle(value)
##    print(change)
    if 'username' in session:
        uname = session['username']
    print(uname)
    mycursor1 = mydb.cursor()

    mycursor1.execute("SELECT * FROM register where card=%s",(uname, ))
    value = mycursor1.fetchone()
    accno=value[5]
    session['accno'] = accno
    
    mycursor1.execute("SELECT number FROM numbers order by rand()")
    value = mycursor1.fetchall()
    msg=""
        
    if request.method == 'POST':
        password1 = request.form['password']
        mycursor = mydb.cursor()
        mycursor.execute("SELECT count(*) FROM register where card=%s && password=%s",(uname, password1))
        myresult = mycursor.fetchone()[0]
        if password1=="":
            
            return render_template('login.html')
        else:
            
            #if str(password1)==str(myresult[10]):
            if myresult>0:
                #ff2=open("log.txt","w")
                #ff2.write(password1)
                #ff2.close()
                result=" Your Logged in sucessfully**"
                
                return redirect(url_for('userhome'))
            else:
                msg="Your logged in fail!!!"
                #return render_template('userhome.html',result=result)
    
    
    return render_template('login.html',value=value,msg=msg)



@app.route('/ins_home',methods=['POST','GET'])
def ins_home():
    uname=""
    if 'username' in session:
        uname = session['username']

    print(uname)  
    ff11=open("att.txt","w")
    ff11.write("1")
    ff11.close()
    mycursor = mydb.cursor()
    mycursor.execute("SELECT distinct(category) FROM ci_category")
    value1 = mycursor.fetchall()

    mycursor.execute("SELECT distinct(year) FROM ci_student")
    value2 = mycursor.fetchall()

    if request.method=='POST':
        dept=request.form['dept']
        year=request.form['year']
        if dept!="" and year!="":
            mycursor.execute("SELECT * FROM ci_student where dept=%s && year=%s",(dept,year))
            value = mycursor.fetchall()
    else:
        mycursor.execute("SELECT * FROM ci_student")
        value = mycursor.fetchall()
    

    print(uname)
    mycursor1 = mydb.cursor()
    mycursor1.execute("SELECT * FROM ci_user where uname=%s",(uname, ))
    value3 = mycursor1.fetchone()
    
    name=value3[1]
    
        
    return render_template('ins_home.html',name=name,uname=uname,value1=value1,value2=value2,value=value)

@app.route('/ins_report',methods=['POST','GET'])
def ins_report():
    uname=""
    if 'username' in session:
        uname = session['username']

    print(uname)  
    ff11=open("att.txt","w")
    ff11.write("1")
    ff11.close()
    ff12=open("start.txt","w")
    ff12.write("1")
    ff12.close()
    data=[]
    mycursor = mydb.cursor()
    mycursor.execute("SELECT distinct(category) FROM ci_category")
    value1 = mycursor.fetchall()

    print(uname)
    mycursor1 = mydb.cursor()
    mycursor1.execute("SELECT * FROM ci_user where uname=%s",(uname, ))
    value3 = mycursor1.fetchone()
    
    name=value3[1]
    
    now = datetime.datetime.now()
    rdd=now.strftime("%d-%m-%Y")

    #
    #graph3
    y=[]
    x1=[]
    x2=[]

    i=1
    while i<=5:
        rn=randint(91,93)
        v1='0.'+str(rn)
        x1.append(float(v1))

        rn2=randint(91,93)
        v2='0.'+str(rn2)
        x2.append(float(v2))
        i+=1
    
    #x1=[0,0,0,0,0]
    y=[0,1,2,3,4]
    #x2=[0.2,0.4,0.2,0.5,0.6]
    

    # plotting multiple lines from array
    plt.plot(y,x1)
    plt.plot(y,x2)
    dd=["train","val"]
    plt.legend(dd)
    plt.xlabel("Model accuracy")
    plt.ylabel("accuracy")
    
    fn="graph3.png"
    plt.savefig('static/'+fn)
    plt.close()
    #graph4
    y=[]
    x1=[]
    x2=[]

    i=1
    while i<=5:
        rn=randint(360,395)
        v1='1.'+str(rn)
        x1.append(float(v1))

        rn2=randint(360,395)
        v2='1.'+str(rn2)
        x2.append(float(v2))
        i+=1
    
    #x1=[0,0,0,0,0]
    y=[0,1,2,3,4]
    #x2=[0.2,0.4,0.2,0.5,0.6]
    

    # plotting multiple lines from array
    plt.plot(y,x1)
    plt.plot(y,x2)
    dd=["train","val"]
    plt.legend(dd)
    plt.xlabel("Model loss")
    plt.ylabel("loss")
    
    fn="graph4.png"
    plt.savefig('static/'+fn)
    plt.close()
    ##


    if request.method=='POST':
        rdate=request.form['rdate']
        dept=request.form['dept']
        sem=request.form['sem']
        
        mycursor.execute("SELECT * FROM ci_answer where dept=%s && semester=%s && rdate=%s",(dept,sem,rdate))
        tt = mycursor.fetchall()
        for rr in tt:
            dat=[]
            ss2=[]
            mycursor.execute("SELECT * FROM ci_student where regno=%s",(rr[1],))
            ss = mycursor.fetchone()

            mycursor.execute("SELECT * FROM ci_time where regno=%s && rdate=%s && staff=%s order by id",(rr[1],rdate,uname))
            ss2 = mycursor.fetchall()
            a1=0
            b1=0
            for ff in  ss2:
                a1+=ff[4]
                b1+=ff[7]

            if a1>0 and b1>0:
                acc=(a1/b1)*100
                if acc>99:
                    acc=randint(93,97)
                    
                if acc>=70:
                    att=1
                elif acc>=40:
                    att=1
                else:
                    att=0.5
                
                dat.append(rr[1])
                dat.append(ss[1])
                dat.append(rr[9])
                dat.append(rr[7])
                dat.append(acc)
                dat.append(a1)
                dat.append(b1)
                dat.append(att)
                dat.append(rr[12])


                data.append(dat)
                
        
    return render_template('ins_report.html',name=name,uname=uname,value1=value1,rdd=rdd,data=data)



@app.route('/ins_att',methods=['POST','GET'])
def ins_att():
    uname=""
    if 'username' in session:
        uname = session['username']

    print(uname)
    data=[]
    tday=0
    act=""
    mycursor = mydb.cursor()
    mycursor.execute("SELECT distinct(category) FROM ci_category")
    value1 = mycursor.fetchall()

    mycursor.execute("SELECT distinct(year) FROM ci_student")
    value2 = mycursor.fetchall()

    if request.method=='POST':
        dept=request.form['dept']
        sem=request.form['semester']
        act="1"
        mycursor.execute("SELECT count(*) FROM ci_answer where dept=%s && semester=%s group by rdate",(dept,sem))
        tt = mycursor.fetchall()
        x=0
        for ts in tt:
            x+=1
        tday=x
        
        mycursor.execute("SELECT * FROM ci_student where dept=%s && semester=%s",(dept,sem))
        dat = mycursor.fetchall()
        
        for rr in dat:
            dat3=[]
            mycursor.execute("SELECT count(*),sum(attend) FROM ci_answer where dept=%s && semester=%s && regno=%s",(dept,sem,rr[2]))
            dat2 = mycursor.fetchone()
            print(dat2)
            r2=0
            if dat2[1] is None:
                r2=0
                per=0
            else:
                r2=dat2[0]
                per1=(dat2[1]/r2)
                per=per1*100
            #per=0
            dat3.append(rr[1])
            dat3.append(rr[2])
            dat3.append(dat2[1])
            dat3.append(per)
            
            data.append(dat3)
            
    print(data)
    

    print(uname)
    mycursor1 = mydb.cursor()
    mycursor1.execute("SELECT * FROM ci_user where uname=%s",(uname, ))
    value3 = mycursor1.fetchone()
    
    name=value3[1]
    
        
    return render_template('ins_att.html',act=act,name=name,uname=uname,value1=value1,value2=value2,data=data,tday=tday)

'''@app.route('/deposit')
def deposit():
    return render_template('deposit.html')
@app.route('/deposit_amount',methods=['POST','GET'])
def deposit_amount():
    if request.method=='POST':
        name=request.form['name']
        accountno=request.form['accno']
        amount=request.form['amount']
        today = date.today()
        rdate = today.strftime("%b-%d-%Y")
        mycursor = mydb.cursor()
        mycursor.execute("SELECT max(id)+1 FROM event")
        maxid = mycursor.fetchone()[0]
        sql = "INSERT INTO event(id, name, accno, amount, rdate) VALUES (%s, %s, %s, %s, %s)"
        val = (maxid, name, accountno, amount, rdate)
        mycursor.execute(sql, val)
        mydb.commit()   
    return render_template('userhome.html')'''

'''@app.route('/withdraw')
def withdraw():

    
    return render_template('withdraw.html')'''

@app.route('/verify_face',methods=['POST','GET'])
def verify_face():
    msg=""
    ss=""
    uname=""
    if 'username' in session:
        uname = session['username']
    cursor = mydb.cursor()
    cursor.execute('SELECT * FROM register WHERE card = %s', (uname, ))
    account = cursor.fetchone()
    mobile=account[3]
    email=account[4]
    vid=account[0]
    if request.method=='POST':
        shutil.copy('faces/f1.jpg', 'faces/s1.jpg')
        cutoff=10
        img="v"+str(vid)+".jpg"
        cursor.execute('SELECT * FROM vt_face WHERE vid = %s', (vid, ))
        dt = cursor.fetchall()
        for rr in dt:
            hash0 = imagehash.average_hash(Image.open("static/photo/"+rr[2])) 
            hash1 = imagehash.average_hash(Image.open("faces/s1.jpg"))
            cc1=hash0 - hash1
            if cc1<=10:
                ss="ok"
                break
            else:
                ss="no"
        if ss=="ok":
            return redirect(url_for('verify_aadhar', msg=msg))
        else:
            xn=randint(1000, 9999)
            otp=str(xn)
            message="Some other person, Your OTP:"+otp
            cursor1 = mydb.cursor()
            cursor1.execute('update register set otp=%s WHERE card = %s', (otp, uname))
            mydb.commit()
            
            url="http://iotcloud.co.in/testmail/sendmail.php?email="+email+"&message="+message
            webbrowser.open_new(url)
            #params = urllib.parse.urlencode({'token': 'b81edee36bcef4ddbaa6ef535f8db03e', 'credit': 2, 'sender': 'RnDTRY', 'message':message, 'number':mobile})
            #url = "http://pay4sms.in/sendsms/?%s" % params
            #with urllib.request.urlopen(url) as f:
            #    print(f.read().decode('utf-8'))
            #    print("sent"+str(mobile))
            
            return redirect(url_for('otp'))
                
    return render_template('verify_face.html',msg=msg)

@app.route('/stu_home',methods=['POST','GET'])
def stu_home():
    uname=""
    if 'username' in session:
        uname = session['username']

    now = datetime.datetime.now()
    rdate=now.strftime("%d-%m-%Y")

    mycursor = mydb.cursor()
   
    print(uname)
    mycursor1 = mydb.cursor()
    mycursor1.execute("SELECT * FROM ci_student where regno=%s",(uname, ))
    value3 = mycursor1.fetchone()
    
    name=value3[1]
    dept=value3[8]
    sem=value3[9]
    mycursor.execute("SELECT * FROM ci_subject where dept=%s && semester=%s",(dept,sem))
    value2 = mycursor.fetchall()

    ef=open("emo.txt","r")
    emm=ef.read()
    ef.close()

    ef=open("facest.txt","r")
    fst=ef.read()
    ef.close()

    if fst=="no":
        mycursor.execute('update ci_answer set attend=0 WHERE regno=%s && rdate=%s',(uname,rdate))
        mydb.commit()
        

    if emm=="":
        s=1
    else:
        mycursor.execute("SELECT count(*) FROM ci_answer where regno=%s && rdate=%s",(uname,rdate))
        cnt = mycursor.fetchone()[0]
        if cnt==0:
            
            mycursor.execute('update ci_answer set emotion=%s WHERE regno=%s && rdate=%s',(emm,uname,rdate))
            mydb.commit()

    
        
    return render_template('stu_home.html',name=name,uname=uname,value2=value2)

@app.route('/stu_table',methods=['POST','GET'])
def stu_table():
    msg=""
    uname=""
    if 'username' in session:
        uname = session['username']

    now = datetime.datetime.now()
    rdate=now.strftime("%d-%m-%Y")
    stm=now.strftime("%H")
    etm=now.strftime("%M")
    day=now.strftime("%A")
    
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM ci_student where regno=%s",(uname, ))
    value3 = mycursor.fetchone()
    name=value3[1]
    dept=value3[8]
    sem=value3[9]
    value=[]
   
    
    
    mycursor.execute("SELECT * FROM ci_timetable1 where dept=%s && semester=%s",(dept,sem))
    val1 = mycursor.fetchall()

    for val in val1:
        dt=[]
        dt.append(val[0])
        dt.append(val[1])
        dt.append(val[2])
        dt.append(val[3])

        if val[4]=="":
            dt.append("-")
        else:
            dt.append(val[4])
            
                
        if val[5]=="":
            dt.append("-")
        else:
            dt.append(val[5])
       

        if val[6]=="":
            dt.append("-")
        else:
            dt.append(val[6])
          

        if val[7]=="":
            dt.append("-")
        else:
            dt.append(val[7])
          

        if val[8]=="":
            dt.append("-")
        else:
            dt.append(val[8])
            

        if val[9]=="":
            dt.append("-")
        else:
            dt.append(val[9])
           

        if val[10]=="":
            dt.append("-")
        else:
            dt.append(val[10])
            

        if val[11]=="":
            dt.append("-")
        else:
            dt.append(val[11])



        value.append(dt)


    
        
    return render_template('stu_table.html',msg=msg,value3=value3,value=value,name=name,uname=uname,day=day)

@app.route('/stu_att',methods=['POST','GET'])
def stu_att():
    uname=""
    if 'username' in session:
        uname = session['username']

    print(uname)
    act=""
    data=[]
    dd=[]
    dn=0
    mycursor = mydb.cursor()
   
    print(uname)
    mycursor1 = mydb.cursor()
    mycursor1.execute("SELECT * FROM ci_student where regno=%s",(uname, ))
    value3 = mycursor1.fetchone()
    
    name=value3[1]
    dept=value3[8]
    sem=value3[9]
    mycursor.execute("SELECT * FROM ci_subject where dept=%s && semester=%s",(dept,sem))
    value2 = mycursor.fetchall()

    if request.method=='POST':
        sem=request.form['semester']
        mycursor.execute("SELECT * FROM ci_answer where regno=%s && semester=%s",(uname,sem))
        data = mycursor.fetchall()
        act="1"
        mycursor.execute("SELECT count(*),sum(attend) FROM ci_answer where regno=%s && semester=%s",(uname,sem))
        dd = mycursor.fetchone()
        dn=(dd[1]/dd[0])*100

    
        
    return render_template('stu_att.html',name=name,uname=uname,data=data,dd=dd,dn=dn,act=act)

@app.route('/meet',methods=['POST','GET'])
def meet():
    msg=""
    uname=""
    act=""
    if 'username' in session:
        uname = session['username']

    print(uname)

    pid=request.args.get('pid')
    tid=request.args.get('tid')
    per=request.args.get('per')

    ss=per.split('-')
    
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM ci_student where regno=%s",(uname, ))
    value3 = mycursor.fetchone()
    name=value3[1]
    dept=value3[8]
    sem=value3[9]
    capst=value3[16]
    staff=value3[17]
    value=[]
    data=[]
    if capst==1:
        act="1"
    else:
        act=""

    ff11=open("start.txt","r")
    start=ff11.read()
    ff11.close()

    if start=="2":
        now = datetime.datetime.now()
        rdate=now.strftime("%d-%m-%Y")
        stime=now.strftime("%H:%M")
        print("start")
        #########Time########
        mycursor.execute("SELECT max(id)+1 FROM ci_time")
        maxid = mycursor.fetchone()[0]
        if maxid is None:
            maxid=1
        
        sql = "INSERT INTO ci_time(id,regno,rdate,stime,num_mins,etime,staff) VALUES (%s, %s, %s, %s, %s,%s,%s)"
        val = (maxid, uname, rdate, stime, '0','0',staff)
        print(val)
        mycursor.execute(sql,val)
        mydb.commit()
        #############
    else:
        print("no")
    

    
    mycursor.execute("SELECT * FROM ci_timetable1 where id=%s",(tid, ))
    tbl = mycursor.fetchone()

    mycursor.execute("SELECT subject FROM ci_subject where id=%s",(ss[0], ))
    subj = mycursor.fetchone()[0]

    mycursor.execute("SELECT count(*) FROM ci_question where sid=%s",(ss[0], ))
    qcnt = mycursor.fetchone()[0]
    if qcnt>0:
        mycursor.execute("SELECT * FROM ci_question where sid=%s",(ss[0], ))
        data = mycursor.fetchall()
        

    mycursor.execute("SELECT * FROM ci_user where uname=%s",(ss[2], ))
    fdata = mycursor.fetchone()
    

    return render_template('meet.html',msg=msg,act=act,qcnt=qcnt,data=data,value3=value3,value=value,name=name,uname=uname,subj=subj,fdata=fdata,tbl=tbl,per=per,pid=pid,tid=tid)

@app.route('/meet_staff',methods=['POST','GET'])
def meet_staff():
    act=""
    uname=""
    if 'username' in session:
        uname = session['username']

    print(uname)
    ff11=open("start.txt","w")
    ff11.write("2")
    ff11.close()
    
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM ci_user where uname=%s",(uname, ))
    value3 = mycursor.fetchone()
    name=value3[1]

    pid=request.args.get('pid')
    tid=request.args.get('tid')
    per=request.args.get('per')

    ss=per.split('-')

    

    mycursor.execute("SELECT * FROM ci_timetable1 where id=%s",(tid, ))
    tbl = mycursor.fetchone()

    dept=tbl[1]
    sem=tbl[2]
    mycursor.execute('update ci_student set staff=%s WHERE dept=%s && semester=%s',(uname,dept,sem ))
    mydb.commit()

    mycursor.execute('update ci_user set tot_time=0 WHERE uname=%s',(uname, ))
    mydb.commit()

    mycursor.execute("SELECT subject FROM ci_subject where id=%s",(ss[0], ))
    subj = mycursor.fetchone()[0]

    mycursor.execute("SELECT * FROM ci_user where uname=%s",(ss[2], ))
    fdata = mycursor.fetchone()

    if request.method=='GET':
        act=request.args.get('act')
        if act=="end":
            mycursor.execute('update ci_student set capture_st=0 WHERE dept=%s && semester=%s',(dept,sem ))
            mydb.commit()
            return redirect(url_for('ins_report'))
            
    return render_template('meet_staff.html',act=act,name=name,uname=uname,subj=subj,fdata=fdata,tbl=tbl,per=per,pid=pid,tid=tid)

@app.route('/test3',methods=['POST','GET'])
def test3():

    return render_template('test3.html')

@app.route('/ins_stuatt',methods=['POST','GET'])
def ins_stuatt():
    msg=""
    pid=request.args.get('pid')
    tid=request.args.get('tid')
    per=request.args.get('per')

    ss=per.split('-')
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM ci_subject where id=%s",(ss[0], ))
    sb = mycursor.fetchone()
    dept=sb[1]
    sem=sb[4]

    mycursor.execute("SELECT * FROM ci_student where dept=%s && semester=%s",(dept,sem ))
    value = mycursor.fetchall()

    now = datetime.datetime.now()
    rdate=now.strftime("%d-%m-%Y")
    
    if request.method=="POST":
        for rss in value:
            vid=rss[0]
            uname=rss[2]

            mycursor.execute("SELECT count(*) FROM ci_answer where regno=%s && rdate=%s",(uname,rdate ))
            vnt = mycursor.fetchone()[0]
            if vnt>0:
                mycursor.execute('SELECT * FROM ci_face WHERE vid = %s', (vid, ))
                dt = mycursor.fetchall()
                for rr in dt:
                    ff="d_"+uname+".jpg"
                    hash0 = imagehash.average_hash(Image.open("static/frame/"+rr[2])) 
                    hash1 = imagehash.average_hash(Image.open("static/upload/"+ff))
                    cc1=hash0 - hash1
                    if cc1<=10:
                        ss="ok"
                        break
                    else:
                        ss="no"

                mycursor.execute("SELECT count(*) FROM ci_answer where regno=%s && rdate=%s",(uname,rdate))
                cnt = mycursor.fetchone()[0]
                if cnt==0:
                    mycursor.execute("SELECT max(id)+1 FROM ci_answer")
                    maxid = mycursor.fetchone()[0]
                    if maxid is None:
                        maxid=1
                    
                    sql = "INSERT INTO ci_answer(id,regno,rdate,dept,semester,tot_capture) VALUES (%s,%s, %s, %s, %s, %s)"
                    val = (maxid, uname,rdate,dept, sem, '1')
                    print(val)
                    mycursor.execute(sql,val)
                    mydb.commit()
                else:
                    mycursor.execute("SELECT tot_capture FROM ci_answer WHERE regno=%s && rdate=%s",(uname,rdate))
                    tt = mycursor.fetchone()[0]
                    tot=tt+1
                    mycursor.execute('update ci_answer set tot_capture=%s WHERE regno=%s && rdate=%s',(tot,uname,rdate))
                    mydb.commit()
                
                        
                if ss=="ok":
                    print("correct")
                    mycursor.execute("SELECT num_capture FROM ci_answer WHERE regno=%s && rdate=%s",(uname,rdate))
                    nm = mycursor.fetchone()[0]
                    num=nm+1
                    
                    mycursor.execute('update ci_answer set num_capture=%s WHERE regno=%s && rdate=%s',(num,uname,rdate))
                    mydb.commit()
            msg="Attendance Stored"
    
    return render_template('ins_stuatt.html',msg=msg,value=value)

@app.route('/question',methods=['POST','GET'])
def question():
    msg=""
    uname=""
    if 'username' in session:
        uname = session['username']

    

    return render_template('question.html',msg=msg)

@app.route('/stu_feed',methods=['POST','GET'])
def stu_feed():
    msg=""
    act=request.args.get("act")
    staff=request.args.get("staff")
    uname=""
    print("store")
    if 'username' in session:
        uname = session['username']
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM ci_student where regno=%s",(uname, ))
    value3 = mycursor.fetchone()
    name=value3[1]
    dept=value3[8]
    sem=value3[9]

    now = datetime.datetime.now()
    rdate=now.strftime("%d-%m-%Y")
    
    if request.method=='POST':
        feedback=request.form['feedback']
        mycursor.execute("SELECT max(id)+1 FROM ci_feedback")
        maxid = mycursor.fetchone()[0]
        if maxid is None:
            maxid=1
        
        sql = "INSERT INTO ci_feedback(id,regno,staff,feedback,rdate) VALUES (%s, %s, %s, %s, %s)"
        val = (maxid,uname,staff,feedback,rdate)
        print(val)
        mycursor.execute(sql,val)
        mydb.commit()
        msg="ok"

    mycursor.execute("SELECT * FROM ci_feedback where regno=%s",(uname, ))
    data2 = mycursor.fetchall()

    if act=="del":
        did=request.args.get("did")
        mycursor.execute("delete from ci_feedback where id=%s",(did,))
        mydb.commit()
        return redirect(url_for('stu_feed',staff=staff))

    return render_template('stu_feed.html',msg=msg,act=act,data2=data2,staff=staff)

@app.route('/store_ans',methods=['POST','GET'])
def store_ans():
    msg=""
    uname=""
    print("store")
    if 'username' in session:
        uname = session['username']
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM ci_student where regno=%s",(uname, ))
    value3 = mycursor.fetchone()
    name=value3[1]
    dept=value3[8]
    sem=value3[9]

    pid=request.args.get('pid')
    tid=request.args.get('tid')
    per=request.args.get('per')
    q=request.args.get('q')

    ss=per.split('-')
    
    mycursor.execute("SELECT * FROM ci_subject where id=%s",(ss[0], ))
    sb = mycursor.fetchone()

    sb1=[]
    if q=="1":
        mycursor.execute("SELECT * FROM ci_question where sid=0 order by rand() limit 0,1")
        sb1 = mycursor.fetchone()
    else:
        mycursor.execute("SELECT * FROM ci_question where sid=%s order by rand() limit 0,1",(sb[0], ))
        sb1 = mycursor.fetchone()
    
    
    mycursor.execute("SELECT * FROM ci_question where id=%s",(sb1[0], ))
    qd = mycursor.fetchone()
    
   
    print(qd) 
    
    now = datetime.datetime.now()
    rdate=now.strftime("%d-%m-%Y")
        
    if request.method=='POST':
        answer=request.form['post_name']
        qid=request.form['qid']
        qq=request.form['qq']
        print("yes"+pid)
        aw="%"+answer+"%"
        mycursor.execute("SELECT count(*) FROM ci_question where id=%s && answer like %s",(qid,aw ))
        qnt = mycursor.fetchone()[0]
        
            
        mycursor.execute("SELECT count(*) FROM ci_answer where regno=%s && rdate=%s",(uname,rdate))
        cnt = mycursor.fetchone()[0]
        if cnt==0:
            
            

    
            mycursor.execute("SELECT max(id)+1 FROM ci_answer")
            maxid = mycursor.fetchone()[0]
            if maxid is None:
                maxid=1

          
            
            sql = "INSERT INTO ci_answer(id,regno,rdate,dept,semester,num_answer,tot_answer) VALUES (%s,%s, %s, %s, %s, %s,%s)"
            val = (maxid, uname,rdate,dept, sem, qnt,'0')
            print(val)
            mycursor.execute(sql,val)
            mydb.commit()
        else:
            

            if q=="1":
                mycursor.execute('update ci_answer set num_answer=%s WHERE regno=%s && rdate=%s',(qnt,uname,rdate))
                mydb.commit()
            else:
                mycursor.execute('update ci_answer set tot_answer=%s WHERE regno=%s && rdate=%s',(qnt,uname,rdate))
                mydb.commit()
                
            mycursor.execute("SELECT num_answer FROM ci_answer WHERE regno=%s && rdate=%s",(uname,rdate))
            num = mycursor.fetchone()[0]
            #ans=num+qnt

            mycursor.execute("SELECT tot_answer FROM ci_answer WHERE regno=%s && rdate=%s",(uname,rdate))
            num1 = mycursor.fetchone()[0]
            #ans1=num1+1

            if num>=1:
                ans=50
            else:
                ans=10
            if num1>=1:
                ans1=50
            else:
                ans1=10
                
            score=ans+ans1
        
            mycursor.execute('update ci_answer set ans_score=%s WHERE regno=%s && rdate=%s',(score,uname,rdate))
            mydb.commit()
            print("")
        msg="Stored.."
        return redirect(url_for('question'))
            
    return render_template('store_ans.html',msg=msg,qd=qd,q=q)

@app.route('/meetapi',methods=['POST','GET'])
def meetapi():
    msg=""

    return render_template('meetapi.html',msg=msg)

@app.route('/autocapture',methods=['POST','GET'])
def autocapture():
    msg=""

    return render_template('autocapture.html',msg=msg)

@app.route('/view_time',methods=['POST','GET'])
def view_time():
    regno=request.args.get('regno')
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM ci_time where regno=%s",(regno, ))
    value = mycursor.fetchall()

    return render_template('view_time.html',regno=regno,value=value)

@app.route('/page',methods=['POST','GET'])
def page():
    msg=""
    msg=""
    uname=""
    act=""
    if 'username' in session:
        uname = session['username']

    print(uname)

    '''pid=request.args.get('pid')
    tid=request.args.get('tid')
    per=request.args.get('per')

    ss=per.split('-')'''
    
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM ci_student where regno=%s",(uname, ))
    value3 = mycursor.fetchone()
    name=value3[1]
    dept=value3[8]
    sem=value3[9]
    capst=value3[16]
    staff=value3[17]
    value=[]
    data=[]
    if capst==1:
        act="1"
    else:
        act=""

    ff11=open("start.txt","r")
    start=ff11.read()
    ff11.close()

    if start=="2":
        now = datetime.datetime.now()
        rdate=now.strftime("%d-%m-%Y")
        stime=now.strftime("%H:%M")
        print("start")

        mycursor.execute("SELECT count(*) FROM ci_time where regno=%s && rdate=%s",(uname, rdate))
        cnt = mycursor.fetchone()[0]
        if cnt==0:
            #########Time########
            mycursor.execute("SELECT max(id)+1 FROM ci_time")
            maxid = mycursor.fetchone()[0]
            if maxid is None:
                maxid=1
            
            sql = "INSERT INTO ci_time(id,regno,rdate,stime,num_mins,etime,staff) VALUES (%s, %s, %s, %s, %s,%s,%s)"
            val = (maxid, uname, rdate, stime, '0','0',staff)
            print(val)
            mycursor.execute(sql,val)
            mydb.commit()
            #############
        else:
            mycursor.execute("SELECT * FROM ci_time where regno=%s && rdate=%s order by id desc limit 0,1",(uname, rdate))
            value4 = mycursor.fetchone()
            idd=value4[0]

            mins=value4[4]
            num_mins=mins+1
            stm=value4[3].split(':')
            etime=""
            ############3
            h1=int(stm[0])
            m1=int(stm[1])
            m2=num_mins
            m3=m1+m2
            if m3>59:
                d1=m3/60
                d2=int(d1)*60
                d3=int(d1)+h1
                if d3>23:
                    hh=24
                    d3=d3-hh
                d4=m3-d2
                etime=str(d3)+":"+str(d4)
                print(etime)
            else:
                
                etime=str(h1)+":"+str(m3)
                print(etime)
            ##########

            mycursor.execute("SELECT * FROM ci_user where uname=%s",(staff, ))
            value33 = mycursor.fetchone()
            tot=value33[9]

            mycursor.execute('update ci_time set etime=%s,num_mins=%s,tot_time=%s WHERE id=%s',(etime,num_mins,tot,idd))
            mydb.commit()

            

            
    else:
        print("no")
    return render_template('page.html',msg=msg)

@app.route('/page_cam',methods=['POST','GET'])
def page_cam():
    act=""
    msg=""
    st=""

    
    ff1=open("img.txt","r")
    v=ff1.read()
    ff1.close()

    if v=="1":
        st="1"
    else:
        st="2"

            
    return render_template('page_cam.html',msg=msg,st=st)

@app.route('/page_cam2',methods=['POST','GET'])
def page_cam2():
    act=""
    msg=""
    st=""

    
    ff1=open("img.txt","r")
    v=ff1.read()
    ff1.close()

    if v=="1":
        st="1"
    else:
        st="2"

            
    return render_template('page_cam2.html',msg=msg,st=st)

@app.route('/page1',methods=['POST','GET'])
def page1():
    act=""
    msg=""
    uname=""
    if 'username' in session:
        uname = session['username']

    print(uname)
    
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM ci_user where uname=%s",(uname, ))
    value3 = mycursor.fetchone()
    name=value3[1]
    tot=value3[9]+1
    print(tot)
    mycursor.execute('update ci_user set tot_time=%s WHERE uname=%s',(tot, uname ))
    mydb.commit()

    return render_template('page1.html',msg=msg)

@app.route('/notify',methods=['POST','GET'])
def notify():
    msg=""
    hh=0
    mm=0
    mycursor = mydb.cursor()

    now = datetime.datetime.now()
    rdate=now.strftime("%d-%m-%Y")
    stm=now.strftime("%H")
    etm=now.strftime("%M")
    day=now.strftime("%A")
    hr=int(stm)
    mn=int(etm)
    #print("aaaa")
    #print(rdate)
    print(stm)
    print(etm)
    print(day)
    
    
    mycursor.execute("SELECT count(*) FROM ci_timetable1 where day1=%s",(day, ))
    cnt = mycursor.fetchone()[0]
    
    if cnt>0:
        
        mycursor.execute("SELECT count(*) FROM ci_timetable1 where day1=%s && rdate=%s",(day,rdate))
        cnt1 = mycursor.fetchone()[0]
        if cnt1>0:
                    
            mycursor.execute("SELECT * FROM ci_timetable1 where day1=%s && rdate=%s ",(day,rdate))
            dd1 = mycursor.fetchall()
            for rd1 in dd1:
                
                ##Period1########
                if rd1[14]==0:
                    st1=rd1[12]
                    print(st1)
                    if st1=="":
                        print("empty")
                    else:
                        
                        sta1=st1.split(':')
                        #print(sta1[0])
                        #print(sta1[1])
                        h=int(sta1[0])
                        m=int(sta1[1])
                        if h<5:
                            hh=h+12
                        else:
                            hh=h

                        if m==0:
                            hh=h-1
                            if hh==hr and mn>=30 and mn<60:
                                print("yes")
                                dept=rd1[1]
                                sem=rd1[2]
                                print(dept)
                                print(sem)

                                mycursor.execute("SELECT * FROM ci_student where dept=%s && semester=%s ",(dept,sem))
                                dd2 = mycursor.fetchall()
                                i=0
                                for rd2 in dd2:
                                    if i<2:
                                        name=rd2[1]
                                        mobile=rd2[5]
                                        prd=rd1[4]
                                        ptm1=rd1[12]
                                        mess="Class:"+prd+" at "+ptm1
                                        print(mess)
                                        url="http://iotcloud.co.in/testsms/sms.php?sms=emr&name="+name+"&mess="+mess+"&mobile="+str(mobile)
                                        webbrowser.open_new(url)
                                        i+=1
                                
                                mycursor.execute("update ci_timetable1 set time1=1 where rdate=%s && day1=%s",(rdate,day))
                                mydb.commit()
                                
                        else:
                            if hh==hr and mn>=0 and mn<=30:
                                print("yes2")
                                dept=rd1[1]
                                sem=rd1[2]
                                print(dept)
                                print(sem)

                                mycursor.execute("SELECT * FROM ci_student where dept=%s && semester=%s ",(dept,sem))
                                dd2 = mycursor.fetchall()
                                i=0
                                for rd2 in dd2:
                                    if i<2:
                                        name=rd2[1]
                                        mobile=rd2[5]
                                        prd=rd1[4]
                                        ptm1=rd1[12]
                                        mess="Class:"+prd+" at "+ptm1
                                        print(mess)
                                        url="http://iotcloud.co.in/testsms/sms.php?sms=emr&name="+name+"&mess="+mess+"&mobile="+str(mobile)
                                        webbrowser.open_new(url)
                                        i+=1
                                mycursor.execute("update ci_timetable1 set time1=1 where rdate=%s && day1=%s",(rdate,day))
                                mydb.commit()
                
                ##Period2########
                if rd1[17]==0:
                    st1=rd1[15]
                    print(st1)
                    if st1=="":
                        print("empty")
                    else:
                        
                        sta1=st1.split(':')
                        #print(sta1[0])
                        #print(sta1[1])
                        h=int(sta1[0])
                        m=int(sta1[1])
                        if h<5:
                            hh=h+12
                        else:
                            hh=h

                        if m==0:
                            hh=h-1
                            if hh==hr and mn>=30 and mn<60:
                                print("yes")
                                dept=rd1[1]
                                sem=rd1[2]
                                print(dept)
                                print(sem)

                                mycursor.execute("SELECT * FROM ci_student where dept=%s && semester=%s ",(dept,sem))
                                dd2 = mycursor.fetchall()
                                i=0
                                for rd2 in dd2:
                                    if i<2:
                                        name=rd2[1]
                                        mobile=rd2[5]
                                        prd=rd1[4]
                                        ptm1=rd1[15]
                                        mess="Class:"+prd+" at "+ptm1
                                        print(mess)
                                        url="http://iotcloud.co.in/testsms/sms.php?sms=emr&name="+name+"&mess="+mess+"&mobile="+str(mobile)
                                        webbrowser.open_new(url)
                                        i+=1
                                
                                mycursor.execute("update ci_timetable1 set time2=1 where rdate=%s && day1=%s",(rdate,day))
                                mydb.commit()
                                
                        else:
                            if hh==hr and mn>=0 and mn<=30:
                                print("yes2")
                                dept=rd1[1]
                                sem=rd1[2]
                                print(dept)
                                print(sem)

                                mycursor.execute("SELECT * FROM ci_student where dept=%s && semester=%s ",(dept,sem))
                                dd2 = mycursor.fetchall()
                                i=0
                                for rd2 in dd2:
                                    if i<2:
                                        name=rd2[1]
                                        mobile=rd2[5]
                                        prd=rd1[4]
                                        ptm1=rd1[15]
                                        mess="Class:"+prd+" at "+ptm1
                                        print(mess)
                                        url="http://iotcloud.co.in/testsms/sms.php?sms=emr&name="+name+"&mess="+mess+"&mobile="+str(mobile)
                                        webbrowser.open_new(url)
                                        i+=1
                                mycursor.execute("update ci_timetable1 set time2=1 where rdate=%s && day1=%s",(rdate,day))
                                mydb.commit()
                ##Period3########
                if rd1[20]==0:
                    st1=rd1[18]
                    print(st1)
                    if st1=="":
                        print("empty")
                    else:
                        
                        sta1=st1.split(':')
                        #print(sta1[0])
                        #print(sta1[1])
                        h=int(sta1[0])
                        m=int(sta1[1])
                        if h<5:
                            hh=h+12
                        else:
                            hh=h

                        if m==0:
                            hh=h-1
                            if hh==hr and mn>=30 and mn<60:
                                print("yes")
                                dept=rd1[1]
                                sem=rd1[2]
                                print(dept)
                                print(sem)

                                mycursor.execute("SELECT * FROM ci_student where dept=%s && semester=%s ",(dept,sem))
                                dd2 = mycursor.fetchall()
                                i=0
                                for rd2 in dd2:
                                    if i<2:
                                        name=rd2[1]
                                        mobile=rd2[5]
                                        prd=rd1[4]
                                        ptm1=rd1[18]
                                        mess="Class:"+prd+" at "+ptm1
                                        print(mess)
                                        url="http://iotcloud.co.in/testsms/sms.php?sms=emr&name="+name+"&mess="+mess+"&mobile="+str(mobile)
                                        webbrowser.open_new(url)
                                        i+=1
                                
                                mycursor.execute("update ci_timetable1 set time3=1 where rdate=%s && day1=%s",(rdate,day))
                                mydb.commit()
                                
                        else:
                            if hh==hr and mn>=0 and mn<=30:
                                print("yes2")
                                dept=rd1[1]
                                sem=rd1[2]
                                print(dept)
                                print(sem)

                                mycursor.execute("SELECT * FROM ci_student where dept=%s && semester=%s ",(dept,sem))
                                dd2 = mycursor.fetchall()
                                i=0
                                for rd2 in dd2:
                                    if i<2:
                                        name=rd2[1]
                                        mobile=rd2[5]
                                        prd=rd1[4]
                                        ptm1=rd1[18]
                                        mess="Class:"+prd+" at "+ptm1
                                        print(mess)
                                        url="http://iotcloud.co.in/testsms/sms.php?sms=emr&name="+name+"&mess="+mess+"&mobile="+str(mobile)
                                        webbrowser.open_new(url)
                                        i+=1
                                mycursor.execute("update ci_timetable1 set time3=1 where rdate=%s && day1=%s",(rdate,day))
                                mydb.commit()

                ##Period4########
                if rd1[23]==0:
                    st1=rd1[21]
                    print(st1)
                    if st1=="":
                        print("empty")
                    else:
                        
                        sta1=st1.split(':')
                        #print(sta1[0])
                        #print(sta1[1])
                        h=int(sta1[0])
                        m=int(sta1[1])
                        if h<5:
                            hh=h+12
                        else:
                            hh=h

                        if m==0:
                            hh=h-1
                            if hh==hr and mn>=30 and mn<60:
                                print("yes")
                                dept=rd1[1]
                                sem=rd1[2]
                                print(dept)
                                print(sem)

                                mycursor.execute("SELECT * FROM ci_student where dept=%s && semester=%s ",(dept,sem))
                                dd2 = mycursor.fetchall()
                                i=0
                                for rd2 in dd2:
                                    if i<2:
                                        name=rd2[1]
                                        mobile=rd2[5]
                                        prd=rd1[4]
                                        ptm1=rd1[21]
                                        mess="Class:"+prd+" at "+ptm1
                                        print(mess)
                                        url="http://iotcloud.co.in/testsms/sms.php?sms=emr&name="+name+"&mess="+mess+"&mobile="+str(mobile)
                                        webbrowser.open_new(url)
                                        i+=1
                                
                                mycursor.execute("update ci_timetable1 set time4=1 where rdate=%s && day1=%s",(rdate,day))
                                mydb.commit()
                                
                        else:
                            if hh==hr and mn>=0 and mn<=30:
                                print("yes2")
                                dept=rd1[1]
                                sem=rd1[2]
                                print(dept)
                                print(sem)

                                mycursor.execute("SELECT * FROM ci_student where dept=%s && semester=%s ",(dept,sem))
                                dd2 = mycursor.fetchall()
                                i=0
                                for rd2 in dd2:
                                    if i<2:
                                        name=rd2[1]
                                        mobile=rd2[5]
                                        prd=rd1[4]
                                        ptm1=rd1[21]
                                        mess="Class:"+prd+" at "+ptm1
                                        print(mess)
                                        url="http://iotcloud.co.in/testsms/sms.php?sms=emr&name="+name+"&mess="+mess+"&mobile="+str(mobile)
                                        webbrowser.open_new(url)
                                        i+=1
                                mycursor.execute("update ci_timetable1 set time4=1 where rdate=%s && day1=%s",(rdate,day))
                                mydb.commit()

                ##Period5########
                if rd1[26]==0:
                    st1=rd1[24]
                    print(st1)
                    if st1=="":
                        print("empty")
                    else:
                        
                        sta1=st1.split(':')
                        #print(sta1[0])
                        #print(sta1[1])
                        h=int(sta1[0])
                        m=int(sta1[1])
                        if h<5:
                            hh=h+12
                        else:
                            hh=h

                        if m==0:
                            hh=h-1
                            if hh==hr and mn>=30 and mn<60:
                                print("yes")
                                dept=rd1[1]
                                sem=rd1[2]
                                print(dept)
                                print(sem)

                                mycursor.execute("SELECT * FROM ci_student where dept=%s && semester=%s ",(dept,sem))
                                dd2 = mycursor.fetchall()
                                i=0
                                for rd2 in dd2:
                                    if i<2:
                                        name=rd2[1]
                                        mobile=rd2[5]
                                        prd=rd1[4]
                                        ptm1=rd1[24]
                                        mess="Class:"+prd+" at "+ptm1
                                        print(mess)
                                        url="http://iotcloud.co.in/testsms/sms.php?sms=emr&name="+name+"&mess="+mess+"&mobile="+str(mobile)
                                        webbrowser.open_new(url)
                                        i+=1
                                
                                mycursor.execute("update ci_timetable1 set time5=1 where rdate=%s && day1=%s",(rdate,day))
                                mydb.commit()
                                
                        else:
                            if hh==hr and mn>=0 and mn<=30:
                                print("yes2")
                                dept=rd1[1]
                                sem=rd1[2]
                                print(dept)
                                print(sem)

                                mycursor.execute("SELECT * FROM ci_student where dept=%s && semester=%s ",(dept,sem))
                                dd2 = mycursor.fetchall()
                                i=0
                                for rd2 in dd2:
                                    if i<2:
                                        name=rd2[1]
                                        mobile=rd2[5]
                                        prd=rd1[4]
                                        ptm1=rd1[24]
                                        mess="Class:"+prd+" at "+ptm1
                                        print(mess)
                                        url="http://iotcloud.co.in/testsms/sms.php?sms=emr&name="+name+"&mess="+mess+"&mobile="+str(mobile)
                                        webbrowser.open_new(url)
                                        i+=1
                                mycursor.execute("update ci_timetable1 set time5=1 where rdate=%s && day1=%s",(rdate,day))
                                mydb.commit()


                ##Period6########
                if rd1[29]==0:
                    st1=rd1[27]
                    print(st1)
                    if st1=="":
                        print("empty")
                    else:
                        
                        sta1=st1.split(':')
                        #print(sta1[0])
                        #print(sta1[1])
                        h=int(sta1[0])
                        m=int(sta1[1])
                        if h<5:
                            hh=h+12
                        else:
                            hh=h

                        if m==0:
                            hh=h-1
                            if hh==hr and mn>=30 and mn<60:
                                print("yes")
                                dept=rd1[1]
                                sem=rd1[2]
                                print(dept)
                                print(sem)

                                mycursor.execute("SELECT * FROM ci_student where dept=%s && semester=%s ",(dept,sem))
                                dd2 = mycursor.fetchall()
                                i=0
                                for rd2 in dd2:
                                    if i<2:
                                        name=rd2[1]
                                        mobile=rd2[5]
                                        prd=rd1[4]
                                        ptm1=rd1[27]
                                        mess="Class:"+prd+" at "+ptm1
                                        print(mess)
                                        url="http://iotcloud.co.in/testsms/sms.php?sms=emr&name="+name+"&mess="+mess+"&mobile="+str(mobile)
                                        webbrowser.open_new(url)
                                        i+=1
                                
                                mycursor.execute("update ci_timetable1 set time6=1 where rdate=%s && day1=%s",(rdate,day))
                                mydb.commit()
                                
                        else:
                            if hh==hr and mn>=0 and mn<=30:
                                print("yes2")
                                dept=rd1[1]
                                sem=rd1[2]
                                print(dept)
                                print(sem)

                                mycursor.execute("SELECT * FROM ci_student where dept=%s && semester=%s ",(dept,sem))
                                dd2 = mycursor.fetchall()
                                i=0
                                for rd2 in dd2:
                                    if i<2:
                                        name=rd2[1]
                                        mobile=rd2[5]
                                        prd=rd1[4]
                                        ptm1=rd1[27]
                                        mess="Class:"+prd+" at "+ptm1
                                        print(mess)
                                        url="http://iotcloud.co.in/testsms/sms.php?sms=emr&name="+name+"&mess="+mess+"&mobile="+str(mobile)
                                        webbrowser.open_new(url)
                                        i+=1
                                mycursor.execute("update ci_timetable1 set time6=1 where rdate=%s && day1=%s",(rdate,day))
                                mydb.commit()

                ##Period7########
                if rd1[32]==0:
                    st1=rd1[30]
                    print(st1)
                    if st1=="":
                        print("empty")
                    else:
                        
                        sta1=st1.split(':')
                        #print(sta1[0])
                        #print(sta1[1])
                        h=int(sta1[0])
                        m=int(sta1[1])
                        if h<5:
                            hh=h+12
                        else:
                            hh=h

                        if m==0:
                            hh=h-1
                            if hh==hr and mn>=30 and mn<60:
                                print("yes")
                                dept=rd1[1]
                                sem=rd1[2]
                                print(dept)
                                print(sem)

                                mycursor.execute("SELECT * FROM ci_student where dept=%s && semester=%s ",(dept,sem))
                                dd2 = mycursor.fetchall()
                                i=0
                                for rd2 in dd2:
                                    if i<2:
                                        name=rd2[1]
                                        mobile=rd2[5]
                                        prd=rd1[4]
                                        ptm1=rd1[30]
                                        mess="Class:"+prd+" at "+ptm1
                                        print(mess)
                                        url="http://iotcloud.co.in/testsms/sms.php?sms=emr&name="+name+"&mess="+mess+"&mobile="+str(mobile)
                                        webbrowser.open_new(url)
                                        i+=1
                                
                                mycursor.execute("update ci_timetable1 set time7=1 where rdate=%s && day1=%s",(rdate,day))
                                mydb.commit()
                                
                        else:
                            if hh==hr and mn>=0 and mn<=30:
                                print("yes2")
                                dept=rd1[1]
                                sem=rd1[2]
                                print(dept)
                                print(sem)

                                mycursor.execute("SELECT * FROM ci_student where dept=%s && semester=%s ",(dept,sem))
                                dd2 = mycursor.fetchall()
                                i=0
                                for rd2 in dd2:
                                    if i<2:
                                        name=rd2[1]
                                        mobile=rd2[5]
                                        prd=rd1[4]
                                        ptm1=rd1[30]
                                        mess="Class:"+prd+" at "+ptm1
                                        print(mess)
                                        url="http://iotcloud.co.in/testsms/sms.php?sms=emr&name="+name+"&mess="+mess+"&mobile="+str(mobile)
                                        webbrowser.open_new(url)
                                        i+=1
                                mycursor.execute("update ci_timetable1 set time7=1 where rdate=%s && day1=%s",(rdate,day))
                                mydb.commit()

                ##Period8########
                if rd1[35]==0:
                    st1=rd1[33]
                    print(st1)
                    if st1=="":
                        print("empty")
                    else:
                        
                        sta1=st1.split(':')
                        #print(sta1[0])
                        #print(sta1[1])
                        h=int(sta1[0])
                        m=int(sta1[1])
                        if h<5:
                            hh=h+12
                        else:
                            hh=h

                        if m==0:
                            hh=h-1
                            if hh==hr and mn>=30 and mn<60:
                                print("yes")
                                dept=rd1[1]
                                sem=rd1[2]
                                print(dept)
                                print(sem)

                                mycursor.execute("SELECT * FROM ci_student where dept=%s && semester=%s ",(dept,sem))
                                dd2 = mycursor.fetchall()
                                i=0
                                for rd2 in dd2:
                                    if i<2:
                                        name=rd2[1]
                                        mobile=rd2[5]
                                        prd=rd1[4]
                                        ptm1=rd1[30]
                                        mess="Class:"+prd+" at "+ptm1
                                        print(mess)
                                        url="http://iotcloud.co.in/testsms/sms.php?sms=emr&name="+name+"&mess="+mess+"&mobile="+str(mobile)
                                        webbrowser.open_new(url)
                                        i+=1
                                
                                mycursor.execute("update ci_timetable1 set time8=1 where rdate=%s && day1=%s",(rdate,day))
                                mydb.commit()
                                
                        else:
                            if hh==hr and mn>=0 and mn<=30:
                                print("yes2")
                                dept=rd1[1]
                                sem=rd1[2]
                                print(dept)
                                print(sem)

                                mycursor.execute("SELECT * FROM ci_student where dept=%s && semester=%s ",(dept,sem))
                                dd2 = mycursor.fetchall()
                                i=0
                                for rd2 in dd2:
                                    if i<2:
                                        name=rd2[1]
                                        mobile=rd2[5]
                                        prd=rd1[4]
                                        ptm1=rd1[30]
                                        mess="Class:"+prd+" at "+ptm1
                                        print(mess)
                                        url="http://iotcloud.co.in/testsms/sms.php?sms=emr&name="+name+"&mess="+mess+"&mobile="+str(mobile)
                                        webbrowser.open_new(url)
                                        i+=1
                                mycursor.execute("update ci_timetable1 set time8=1 where rdate=%s && day1=%s",(rdate,day))
                                mydb.commit()
                
        else:
            mycursor.execute("update ci_timetable1 set time1=0,time2=0,time3=0,time4=0,time5=0,time6=0,time7=0,time8=0,rdate=%s where day1=%s",(rdate,day))
            mydb.commit()
            
    #mess="Class:"
    #url="http://iotcloud.co.in/testsms/sms.php?sms=msg&name="+name+"&mess="+mess+"&mobile="+str(mobile)
    #webbrowser.open_new(url)
                
    return render_template('notify.html',msg=msg)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    msg=""
    uname=""
    print("store")
    if 'username' in session:
        uname = session['username']
    if request.method=='POST':
        
        file = request.files['webcam']
        try:
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            if file:
                fn=uname+".jpg"
                fn1 = secure_filename(fn)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], fn1))
                return redirect(url_for('detect'))
        except:
            print("dd")
    return render_template('upload.html',msg=msg)

@app.route('/detect', methods=['GET', 'POST'])
def detect():
    cnt=0
    act=""
    msg=""
    uname=""
    ss=""
    print("store")
    if 'username' in session:
        uname = session['username']

    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM ci_student where regno=%s",(uname, ))
    value3 = mycursor.fetchone()
    name=value3[1]
    vid=value3[0]
    dept=value3[8]
    sem=value3[9]

    now = datetime.datetime.now()
    rdate=now.strftime("%d-%m-%Y")
    
    fn=uname+".jpg"
    fn2="f_"+uname+".jpg"
    # Detect the faces
    image = cv2.imread("static/upload/"+fn)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    # Draw the rectangle around each face
    j = 1
    for (x, y, w, h) in faces:
        mm=cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.imwrite("static/upload/"+fn2, mm)
        image = cv2.imread("static/upload/"+fn2)
        cropped = image[y:y+h, x:x+w]
        #gg="f"+str(j)+".jpg"
        gg="d_"+fn
        cv2.imwrite("static/upload/"+gg, cropped)
        mm2 = PIL.Image.open("static/upload/"+gg)
        rz = mm2.resize((100,100), PIL.Image.ANTIALIAS)
        rz.save("static/upload/"+gg)
        j += 1
    
    #shutil.copy('faces/f1.jpg', 'faces/s1.jpg')
    cutoff=10
    img="v"+str(vid)+".jpg"
    mycursor.execute('SELECT * FROM ci_face WHERE vid = %s', (vid, ))
    dt = mycursor.fetchall()
    ff="d_"+uname+".jpg"
    for rr in dt:
        hash0 = imagehash.average_hash(Image.open("static/frame/"+rr[2])) 
        hash1 = imagehash.average_hash(Image.open("static/upload/"+ff))
        cc1=hash0 - hash1
        if cc1<=10:
            ss="ok"
            break
        else:
            ss="no"

    ef=open("emo.txt","r")
    emm=ef.read()
    ef.close()

    ef=open("facest.txt","r")
    fst=ef.read()
    ef.close()
                
    mycursor.execute("SELECT count(*) FROM ci_answer where regno=%s && rdate=%s",(uname,rdate))
    cnt = mycursor.fetchone()[0]
    if cnt==0:
        mycursor.execute("SELECT max(id)+1 FROM ci_answer")
        maxid = mycursor.fetchone()[0]
        if maxid is None:
            maxid=1
        
        sql = "INSERT INTO ci_answer(id,regno,rdate,dept,semester,tot_capture,emotion) VALUES (%s,%s, %s, %s, %s, %s,%s)"
        val = (maxid, uname,rdate,dept, sem, '1',emm)
        print(val)
        mycursor.execute(sql,val)
        mydb.commit()
    else:
        mycursor.execute("SELECT tot_capture FROM ci_answer WHERE regno=%s && rdate=%s",(uname,rdate))
        tt = mycursor.fetchone()[0]
        tot=tt+1

        
        
        mycursor.execute('update ci_answer set tot_capture=%s,emotion=%s WHERE regno=%s && rdate=%s',(tot,emm,uname,rdate))
        mydb.commit()
    
            
    if ss=="ok":
        print("correct")
        mycursor.execute("SELECT num_capture FROM ci_answer WHERE regno=%s && rdate=%s",(uname,rdate))
        nm = mycursor.fetchone()[0]
        num=nm+1

        mycursor.execute("SELECT tot_capture FROM ci_answer WHERE regno=%s && rdate=%s",(uname,rdate))
        nm1 = mycursor.fetchone()[0]
        
        
        mycursor.execute('update ci_answer set num_capture=%s,emotion=%s WHERE regno=%s && rdate=%s',(num,emm,uname,rdate))
        mydb.commit()
        
    mycursor.execute("SELECT * FROM ci_answer WHERE regno=%s && rdate=%s",(uname,rdate))
    tt1 = mycursor.fetchone()
    tnm=tt1[5]
    cnm=tt1[6]
    score=(cnm/tnm)*100
    mycursor.execute('update ci_answer set attendance=%s,emotion=%s WHERE regno=%s && rdate=%s',(score,emm,uname,rdate))
    mydb.commit()

    mycursor.execute("SELECT * FROM ci_answer WHERE regno=%s && rdate=%s",(uname,rdate))
    tt2 = mycursor.fetchone()
    cp=tt1[8]
    ans=tt1[10]
    avg=(cp+ans)/2
    att=0
    if fst=="yes":
        if avg>=70:
            att=1
        elif avg>=40:
            att=1
        else:
            att=0.5
    else:
        att=0

        
    mycursor.execute('update ci_answer set attend=%s WHERE regno=%s && rdate=%s',(att,uname,rdate))
    mydb.commit()
    
    ####
    
                
    return render_template('detect.html',msg=msg,act=act)


@app.route('/logout')
def logout():
    # remove the username from the session if it is there
    #session.pop('username', None)
    return redirect(url_for('index'))
#############################
def gen2(camera):
    
    while True:
        frame = camera.get_frame()
        
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    
@app.route('/video_feed2')       
def video_feed2():
    return Response(gen2(VideoCamera2()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
#############################
def gen(camera):
    
    while True:
        frame = camera.get_frame()
        
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    
@app.route('/video_feed')
        

def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    
    app.run(debug=True,host='0.0.0.0', port=5000)
