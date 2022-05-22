from PySide2.QtWidgets import *
from PySide2.QtUiTools import *
from PySide2.QtGui import QImage,QPixmap
from PyQt5.QtCore import  QTimer
import pandas as pd
import numpy as np
import datetime
import os
import cv2
from PIL import Image, ImageDraw, ImageFont


clerk_info_folder='clerk_info'          #存放员工信息的文件夹
clerk_img_folder='clerk_img'            #存放员工图片信息的文件夹
model_folder='face_detect_model'        #存放脸部识别的模型
clerk_text=clerk_info_folder+'/info.csv'        #记录文本信息的文件
face_model=model_folder+'/csl.yml'              #面部识别文件 
warningtime=0                           #警报次数

    #返回时间和日期的函数
def time_date(x):   
    if x==0:#时间
        return str(datetime.datetime.now()).split()[1].split('.')[0]    
    else:   #日期
        return str(datetime.datetime.now()).split()[0]

class Stats():

    def __init__(self):
        # 从文件中加载UI定义
        super().__init__()
        # 从 UI 定义中动态 创建一个相应的窗口对象
        # 注意：里面的控件对象也成为窗口对象的属性了
        # 比如 self.ui.button , self.ui.textEdit
        self.ui = QUiLoader().load('main.ui')       #动态载入ui文件
        self.cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)      #打开摄像头
        self.timer=QTimer()     #设置定时器
        self.timer.timeout.connect(self.show)   # 这里调用不能有函数括号，不是单纯的运行函数
        self.timer.start(10)    #更新时间为每10秒     
        self.ui.info_collection.clicked.connect(self.info_collection)    #点击录入人脸
        self.ui.create_model.clicked.connect(self.create_model)          #点击生成模型
        self.ui.sign_in.clicked.connect(self.sign_in)                    #点击签到
        self.ui.sign_out.clicked.connect(self.sign_out)                    #点击签到


    def show(self):
        #这里是显示图片
        ret, self.frame = self.cap.read()
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)     # RGB转BGR，不然直接显示会变蓝
        self.frame = cv2.flip(self.frame, 1)                         #摄像头是和人对立的，将图像左右调换回来正常显示。
        image = QImage(self.frame, self.frame.shape[1], self.frame.shape[0],
               self.frame.strides[0], QImage.Format_RGB888)  #Qlmage的参数（data, width, height, bytesPerLine, format ）图像存储使用8-8-8 24位RGB格式
        self.ui.show_face.setPixmap(QPixmap.fromImage(image)) #把图像设置为背景
        self.ui.show_face.setSizePolicy(QSizePolicy.Ignored,QSizePolicy.Ignored)   #按比例填充
        self.ui.show_face.setScaledContents(True)
        #下面是设置时间
        current_date = time_date(1)
        current_time = time_date(0)
        self.ui.Date_Label.setText(current_date)
        self.ui.Time_Label.setText(current_time)

    def other_window(self):
        from info import face_info_collection
        self.w2=face_info_collection()   #实例化子窗口        #这里必须用self,不能另起变量名，不然会闪退

    def info_collection(self):
        self.clerk_face=cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)      #再转化回来，不然图像会偏蓝色（这个图片是用来存储的，所以还得用opencv的颜色通道）
        name, okPressed = QInputDialog.getText(
                    self.ui,
                    "输入框",
                    "请输入录入人脸的员工名称",
                    QLineEdit.Normal,
                    "")
        while(name.isdigit()):  #这里是经过实践发现填数字的话最后会变成矩阵类型导致无法识别，所以强制填数字
            QMessageBox.warning(self.ui, '警告', '输入姓名格式错误,请重新录入')
            return 0
        if  okPressed:  #确认的话再输入工号
            id, okPressed = QInputDialog.getText(
                    self.ui,
                    "输入框",
                    "请输入录入人脸的员工工号",
                    QLineEdit.Normal,
                    "")
            if okPressed:   #工号也确认了的话，再进行检查：
                #文档读取
                if not os.path.exists(clerk_text):  # 是否存在原纪录,重点就是从哪里追加，和是否读取文件
                    if not os.path.exists(clerk_img_folder):    #这里再判断一次是为了防止有的老六录完信息又退出了，导致有文件夹没内容
                        os.makedirs(clerk_info_folder)      #先创建一个文件夹
                        os.makedirs(clerk_img_folder)
                    info=pd.DataFrame(columns=['name','id'])    #新建一个
                    index=1
                else:
                    info=pd.read_csv(clerk_text)        #有的话就直接读取出来
                    index=info.shape[0]     #获得行数

                #错误检测
                while(not id.isdigit() or eval(id) in info['id'].tolist() ):#防止用户输入非数字或者工号重复
                    if(not id.isdigit()):       
                        QMessageBox.warning(self.ui, '警告', '输入工号格式错误,请重新录入')
                        return 0
                    if(eval(id) in info['id'].tolist()):    
                        QMessageBox.warning(self.ui, '警告', '所输工号已经存在，请重新录入')
                        return 0

                id=eval(id)     #转化成数值类型
                #都没有错误的话就直接录入
                cv2.imencode('.jpg', self.clerk_face)[1].tofile(clerk_img_folder+'/'+str(id)+'.'+name+'.jpg')            #解决了中文问题
                info.loc[index]=[str(name),id]               #追加一行数据
                QMessageBox.information(self.ui, '信息', '人脸信息已录入')
                info.to_csv(clerk_text,index=False)

    def create_model(self):
        if not os.path.exists(clerk_text):
            QMessageBox.warning(self.ui, '警告', '你得先录入人脸才可以训练模型呢')
            return 0
        if os.path.exists(face_model):  # 是否已经有模型
            choice = QMessageBox.question(self.ui, '确认', '已经有训练好的分类器，是否因为人员变动而要重新训练？')
            if choice == QMessageBox.No: 
                return 0                            
        else:      #没有的话还要重新建立文件夹
            os.makedirs(model_folder)       #先创建一个文件夹

        #储存图像以及对应的信息
        face_info,ids=[],[]
        #储存原有图像(也就是把放图像文件夹中的图像路径一个一个读出来)
        img_paths=[os.path.join(clerk_img_folder,f)for f in os.listdir(clerk_img_folder)]
        #加载分类器
        face_detector=cv2.CascadeClassifier(r'C:\Users\csl\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages\cv2\data\haarcascade_frontalface_alt2.xml')
        #遍历列表中的图片
        for img in img_paths:
            #打开图像，并转化成灰度的，PIL库有9中不同的打开模式：1，L,P,RGB....
            PIL_img=Image.open(img).convert('L')
            #把图像信息转化成数组
            img_np=np.array(PIL_img,'uint8')
            #获取特征
            faces=face_detector.detectMultiScale(img_np)
            #获取对应图片的id
            info=int(os.path.split(img)[1].split('.')[0])       #还是没明白为什么这里要int
            #预防无面容图像
            for x,y,w,h in faces:
                ids.append(info)
                face_info.append(img_np[y:y+h,x:x+w])

        recognizer=cv2.face.LBPHFaceRecognizer_create()      #这里会报错因为少装了一个库：pip3 install opencv-contrib-python -i https://pypi.tuna.tsinghua.edu.cn/simple
        #训练识别器
        try:
            recognizer.train(face_info,np.array(ids))
        except:
            QMessageBox.warning(self.ui, '警告', '啊哦，训练出错了，建议重新录入人脸')
        #保存文件
        recognizer.write(face_model)

        QMessageBox.information(self.ui, '信息', '模型已训练好！')

    #面部识别时解决中文乱码问题
    def cv2AddChineseText(self, text, position, textColor=(0, 255, 0), textSize=30):
        if (isinstance(self.frame, np.ndarray)):  # 判断是否OpenCV图片类型
            self.frame = Image.fromarray(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB))
        # 创建一个可以在给定图像上绘图的对象
        draw = ImageDraw.Draw(self.frame)
        # 字体的格式
        fontStyle = ImageFont.truetype(
            "simsun.ttc", textSize, encoding="utf-8")
        # 绘制文本
        draw.text(position, text, textColor, font=fontStyle)
        # 转换回OpenCV格式
        return cv2.cvtColor(np.asarray(self.frame), cv2.COLOR_RGB2BGR)

    def name(self):
        path = clerk_img_folder
        names = []
        imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
        for imagePath in imagePaths:
            # print(os.path.split(imagePath)[1].split('.',2))
            name = str(os.path.split(imagePath)[1].split('.',2)[1])
            names.append(name)
        return names

        #面部识别
    def face_detect_demo(self,recogizer,names,sign):
        gray=cv2.cvtColor(self.frame,cv2.COLOR_BGR2GRAY)#转换为灰度
        face_detector=cv2.CascadeClassifier(r'C:\Users\csl\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages\cv2\data\haarcascade_frontalface_default.xml')
        #face=face_detector.detectMultiScale(gray,1.1,5,cv2.CASCADE_SCALE_IMAGE,(100,100),(300,300))
        face=face_detector.detectMultiScale(gray)
        clerk_name='unkonw'
        res=1       #默认没有识别到用户
        for x,y,w,h in face:        #这个应该只在这个函数里运行一次，而之后主要看更新频率
            cv2.rectangle(self.frame,(x,y),(x+w,y+h),color=(0,0,255),thickness=2)
            cv2.circle(self.frame,center=(x+w//2,y+h//2),radius=w//2,color=(0,255,0),thickness=1)
            # 人脸识别
            ids, confidence = recogizer.predict(gray[y:y + h, x:x + w])
            #print('标签id:',ids,'置信评分：', confidence)
            if confidence > 80:     #评分越大越不可信 
                global warningtime
                warningtime += 1
                if warningtime > 80:       #不符合次数达到一定次数则提醒
                    warningtime = 0         #错误次数重置为0
                    return 2,'unkonw'
                cv2.putText(self.frame, 'unkonw', (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
            if confidence < 30:     #适当调小可以减小误判率
                clerk_name=names[ids-1]
                # result=cv.putText(self.frame,str(names[ids-1]), (x + 10, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)    #把名字打到框图上
                self.frame=self.cv2AddChineseText(str(names[ids-1]),(x + 10, y - 10),(0, 255, 0),30)        #用中文写名字在识别框上
                self.clerk_face=cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)      #再转化回来，不然图像会偏蓝色（这个图片是用来存储的，所以还得用opencv的颜色通道）
                cv2.imencode('.jpg', self.clerk_face)[1].tofile(clerk_name+sign+'.jpg')
                res=0
        return res,clerk_name

    def sign_in(self):
        if not os.path.exists(face_model):  # 是否已经有模型
            QMessageBox.warning(self.ui, '警告', '嗯哼，你得先训练模型呢')
            return 0
        else:
             #加载训练文件
            recogizer=cv2.face.LBPHFaceRecognizer_create()
            recogizer.read(face_model)
            names=self.name()
            def face_sign_in():     #内置函数方便被计时器触发
                res,clerk_name=self.face_detect_demo(recogizer,names,'签到截图')
                if res==0:      #识别到了人脸,开始签到
                    info=pd.read_csv(sign_table)
                    index=info.shape[0]     #获得行数
                    # print(type(clerk_name),[type(i) for i in info['name'].values])      #这里不行，得把后者完全转换为字符串
                    #先判断，后操作
                    choice = QMessageBox.question(self.ui, '确认', '签到人是否为：'+clerk_name)
                    if choice == QMessageBox.No: 
                        self.timer_sign_in.stop()    #计时器停止计时
                        return 0 
                    #后续操作    
                    if clerk_name in info['name'].values:       #是否重复签到
                        QMessageBox.warning(self.ui, '警告', '签到失败，'+clerk_name+'重复签到')
                    else:     
                        info.loc[index]=[clerk_name,time_date(0),'']
                        info.to_csv(sign_table,index=False)
                        QMessageBox.information(self.ui, '信息', clerk_name+'已签到成功！')
                    self.timer_sign_in.stop()    #计时器停止计时
                    self.beauty()
                elif res==1:            #未识别到用户
                    self.ui.info_text.clear()    #先清除之前的内容
                    self.ui.info_text.setText('暂未识别成功，正在尝试。。。')
                else:                   #防止异常用户或者长时间未检测到人脸
                    QMessageBox.warning(self.ui, '警告', '识别错误，请重试！')
                    self.timer_sign_in.stop()    #计时器停止计时
                    self.beauty()
            self.timer_sign_in=QTimer()     #设置定时器
            self.timer_sign_in.timeout.connect(face_sign_in)   # 这里调用不能有函数括号，不是单纯的运行函数
            self.timer_sign_in.start(10)    #更新时间为每10秒  
            self.beauty()

    def sign_out(self):
        if not os.path.exists(face_model):  # 是否已经有模型
            QMessageBox.warning(self.ui, '警告', '嗯哼，你得先训练模型呢')
            return 0
        else:
             #加载训练文件
            recogizer=cv2.face.LBPHFaceRecognizer_create()
            recogizer.read(face_model)
            names=self.name()
            def face_sign_in():     #内置函数方便被计时器触发
                res,clerk_name=self.face_detect_demo(recogizer,names,'签退截图')
                if res==0:      #识别到了人脸,开始签到
                    info=pd.read_csv(sign_table)
                    index=info.shape[0]     #获得行数
                    x=info.loc[info.name==clerk_name,'end_time'].values[0]
                    #先确定签退人信息再进行后面的操作
                    choice = QMessageBox.question(self.ui, '确认', '签退人是否为：'+clerk_name)
                    if choice == QMessageBox.No: 
                        self.timer_sign_in.stop()    #计时器停止计时
                        return 0
                    #开始进行后续操作
                    if clerk_name not in info['name'].values:       #是否重复签到
                        QMessageBox.warning(self.ui, '警告', '签退失败，'+clerk_name+'你还没签到呢，先去签个到吧~')
                    elif x==x:  #不为none值
                        QMessageBox.warning(self.ui, '警告', '签退失败，'+clerk_name+'你已经签退了呢~')
                    else:      
                        info.loc[info.name==clerk_name,'end_time']=time_date(0)     #直接修改了第三值
                        info.to_csv(sign_table,index=False)
                        QMessageBox.information(self.ui, '信息', clerk_name+'已签退成功！')
                    self.timer_sign_in.stop()    #计时器停止计时
                    self.beauty()
                elif res==1:            #未识别到用户
                    self.ui.info_text.clear()    #先清除之前的内容
                    self.ui.info_text.setText('暂未识别成功，正在尝试。。。')
                else:                   #防止异常用户或者长时间未检测到人脸
                    QMessageBox.warning(self.ui, '警告', '识别错误，请重试！')
                    self.timer_sign_in.stop()    #计时器停止计时
                    self.beauty()
            self.timer_sign_in=QTimer()     #设置定时器
            self.timer_sign_in.timeout.connect(face_sign_in)   # 这里调用不能有函数括号，不是单纯的运行函数
            self.timer_sign_in.start(10)    #更新时间为每10秒  
            self.beauty()



    def r(self):
        a = self.ui.camera
        cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)
        while(cap.isOpened()):      #人脸检测循环
             ret, frame = cap.read()
             frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)     # RGB转BGR，不然直接显示会变蓝
             frame = cv2.flip(frame, 1)                         #摄像头是和人对立的，将图像左右调换回来正常显示。
             image = QImage(frame, frame.shape[1], frame.shape[0],
                       frame.strides[0], QImage.Format_RGB888)  #Qlmage的参数（data, width, height, bytesPerLine, format ）图像存储使用8-8-8 24位RGB格式
             a.setPixmap(QPixmap.fromImage(image)) #把图像设置为背景
             a.setSizePolicy(QSizePolicy.Ignored,QSizePolicy.Ignored)   #按比例填充
             a.setScaledContents(True)
             key=cv2.waitKey(0)  #更新的同时等待操作
             if self.ui.close_.isChecked() is True:
                break
        cv2.destroyAllWindows()      #相当于把图相框删掉
        cap.release()               #释放摄像头
        print('sss')

    def beauty(self):
        #美观
        self.ui.info_text.clear()    #先清除之前的内容
        self.ui.info_text.setText('祝您使用愉快~')


if __name__ == '__main__':
    app = QApplication([])
    sign_table=time_date(1)+'.csv'               #考勤表(今天的日期),为什么不放最外面是因为找不到那个函数所以放这里了
    if not os.path.exists(sign_table):#是否有今天的考勤表这样将确保程序在调用时就一定会有今天的考勤表
        #没有考勤表就准备新建一个
        info=pd.DataFrame(columns=['name','start_time','end_time']) #新建一个
        info.to_csv(sign_table,index=False)
    stats = Stats()
    stats.ui.show()
    app.exec_()       #这里是让窗口一直循环下去，除非人为关掉(去掉之后窗口弹一下就关掉了)

