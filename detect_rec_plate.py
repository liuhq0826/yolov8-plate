from datetime import datetime

import torch
import cv2
import numpy as np
import argparse
import copy
import time
import os
from ultralytics.nn.tasks import  attempt_load_weights
from plate_recognition.plate_rec import get_plate_result,init_model,cv_imread
from plate_recognition.double_plate_split_merge import get_split_merge
from fonts.cv_puttext import cv2ImgAddText
import pandas as pd
import glob

def allFilePath(rootPath,allFIleList):# 读取文件夹内的文件，放到list
    fileList = os.listdir(rootPath)
    for temp in fileList:
        if os.path.isfile(os.path.join(rootPath,temp)):
            allFIleList.append(os.path.join(rootPath,temp))
        else:
            allFilePath(os.path.join(rootPath,temp),allFIleList)
            
def four_point_transform(image, pts):                       #透视变换得到车牌小图
    # rect = order_points(pts)
    rect = pts.astype('float32')
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped
            

def letter_box(img,size=(640,640)):  #yolo 前处理 letter_box操作
    h,w,_=img.shape
    r=min(size[0]/h,size[1]/w)
    new_h,new_w=int(h*r),int(w*r)
    new_img = cv2.resize(img,(new_w,new_h))
    left= int((size[1]-new_w)/2)
    top=int((size[0]-new_h)/2)   
    right = size[1]-left-new_w
    bottom=size[0]-top-new_h 
    img =cv2.copyMakeBorder(new_img,top,bottom,left,right,cv2.BORDER_CONSTANT,value=(114,114,114))
    return img,r,left,top

def load_model(weights, device):  #加载yolov8 模型
    model = attempt_load_weights(weights,device=device)  # load FP32 model
    return model    

def xywh2xyxy(det):     #xywh转化为xyxy
    y = det.clone()
    y[:,0]=det[:,0]-det[0:,2]/2
    y[:,1]=det[:,1]-det[0:,3]/2
    y[:,2]=det[:,0]+det[0:,2]/2
    y[:,3]=det[:,1]+det[0:,3]/2
    return y

def my_nums(dets,iou_thresh):  #nms操作
    y = dets.clone()
    y_box_score = y[:,:5]
    index = torch.argsort(y_box_score[:,-1],descending=True)
    keep = []
    while index.size()[0]>0:
        i = index[0].item()
        keep.append(i)
        x1=torch.maximum(y_box_score[i,0],y_box_score[index[1:],0])
        y1=torch.maximum(y_box_score[i,1],y_box_score[index[1:],1])
        x2=torch.minimum(y_box_score[i,2],y_box_score[index[1:],2])
        y2=torch.minimum(y_box_score[i,3],y_box_score[index[1:],3])
        zero_=torch.tensor(0).to(device)
        w=torch.maximum(zero_,x2-x1)
        h=torch.maximum(zero_,y2-y1)
        inter_area = w*h
        nuion_area1 =(y_box_score[i,2]-y_box_score[i,0])*(y_box_score[i,3]-y_box_score[i,1]) #计算交集
        union_area2 =(y_box_score[index[1:],2]-y_box_score[index[1:],0])*(y_box_score[index[1:],3]-y_box_score[index[1:],1])#计算并集

        iou = inter_area/(nuion_area1+union_area2-inter_area)#计算iou
        
        idx = torch.where(iou<=iou_thresh)[0]   #保留iou小于iou_thresh的
        index=index[idx+1]
    return keep


def restore_box(dets,r,left,top):  #坐标还原到原图上

    dets[:,[0,2]]=dets[:,[0,2]]-left
    dets[:,[1,3]]= dets[:,[1,3]]-top
    dets[:,:4]/=r
    # dets[:,5:13]/=r

    return dets
    # pass

def post_processing(prediction,conf,iou_thresh,r,left,top):  #后处理

    prediction = prediction.permute(0,2,1).squeeze(0)
    xc = prediction[:, 4:6].amax(1) > conf  #过滤掉小于conf的框         
    x = prediction[xc]
    if not len(x):
        return []
    boxes = x[:,:4]  #框
    boxes = xywh2xyxy(boxes)  #中心点 宽高 变为 左上 右下两个点
    score,index = torch.max(x[:,4:6],dim=-1,keepdim=True) #找出得分和所属类别
    x = torch.cat((boxes,score,x[:,6:14],index),dim=1)      #重新组合
    
    score = x[:,4]
    keep =my_nums(x,iou_thresh)
    x=x[keep]
    x=restore_box(x,r,left,top)
    return x

def pre_processing(img,opt,device):  #前处理
    img, r,left,top= letter_box(img,(opt.img_size,opt.img_size))
    # print(img.shape)
    img=img[:,:,::-1].transpose((2,0,1)).copy()  #bgr2rgb hwc2chw
    img = torch.from_numpy(img).to(device)
    img = img.float()
    img = img/255.0
    img =img.unsqueeze(0)
    return img ,r,left,top

def det_rec_plate(img,img_ori,detect_model,plate_rec_model):
    result_list=[]
    img,r,left,top = pre_processing(img,opt,device)  #前处理
    predict = detect_model(img)[0]                   
    outputs=post_processing(predict,0.3,0.5,r,left,top) #后处理
    for output in outputs:
        result_dict={}
        output = output.squeeze().cpu().numpy().tolist()
        rect=output[:4]
        rect = [int(x) for x in rect]
        label = output[-1]
        roi_img = img_ori[rect[1]:rect[3],rect[0]:rect[2]]
        if roi_img is None or roi_img.size == 0:
            # print(f"Warning: empty roi_img at rect {rect}")
            continue
        # land_marks=np.array(output[5:13],dtype='int64').reshape(4,2)
        # roi_img = four_point_transform(img_ori,land_marks)   #透视变换得到车牌小图
        if int(label):        #判断是否是双层车牌，是双牌的话进行分割后然后拼接
            roi_img=get_split_merge(roi_img)
        plate_number,rec_prob,plate_color,color_conf=get_plate_result(roi_img,device,plate_rec_model,is_color=True)
        
        result_dict['plate_no']=plate_number   #车牌号
        result_dict['plate_color']=plate_color   #车牌颜色
        result_dict['rect']=rect                      #车牌roi区域
        result_dict['detect_conf']=output[4]              #检测区域得分
        # result_dict['landmarks']=land_marks.tolist() #车牌角点坐标
        # result_dict['rec_conf']=rec_prob   #每个字符的概率
        result_dict['roi_height']=roi_img.shape[0]  #车牌高度
        # result_dict['plate_color']=plate_color
        # if is_color:
        result_dict['color_conf']=color_conf    #颜色得分
        result_dict['plate_type']=int(label)   #单双层 0单层 1双层
        result_list.append(result_dict)
    return result_list


def draw_result(orgimg,dict_list,is_color=False):   # 车牌结果画出来
    result_str =""
    for result in dict_list:
        rect_area = result['rect']
        
        x,y,w,h = rect_area[0],rect_area[1],rect_area[2]-rect_area[0],rect_area[3]-rect_area[1]
        padding_w = 0.05*w
        padding_h = 0.11*h
        rect_area[0]=max(0,int(x-padding_w))
        rect_area[1]=max(0,int(y-padding_h))
        rect_area[2]=min(orgimg.shape[1],int(rect_area[2]+padding_w))
        rect_area[3]=min(orgimg.shape[0],int(rect_area[3]+padding_h))

        height_area = result['roi_height']
        # landmarks=result['landmarks']
        result_p = result['plate_no']
        if result['plate_type']==0:#单层
            result_p+=" "+result['plate_color']
        else:                             #双层
            result_p+=" "+result['plate_color']+"双层"
        result_str+=result_p+" "
        # for i in range(4):  #关键点
        #     cv2.circle(orgimg, (int(landmarks[i][0]), int(landmarks[i][1])), 5, clors[i], -1)
        cv2.rectangle(orgimg,(rect_area[0],rect_area[1]),(rect_area[2],rect_area[3]),(0,0,255),2) #画框
        
        labelSize = cv2.getTextSize(result_p,cv2.FONT_HERSHEY_SIMPLEX,0.5,1) #获得字体的大小
        if rect_area[0]+labelSize[0][0]>orgimg.shape[1]:                 #防止显示的文字越界
            rect_area[0]=int(orgimg.shape[1]-labelSize[0][0])
        orgimg=cv2.rectangle(orgimg,(rect_area[0],int(rect_area[1]-round(1.6*labelSize[0][1]))),(int(rect_area[0]+round(1.2*labelSize[0][0])),rect_area[1]+labelSize[1]),(255,255,255),cv2.FILLED)#画文字框,背景白色
        
        if len(result)>=6:
            orgimg=cv2ImgAddText(orgimg,result_p,rect_area[0],int(rect_area[1]-round(1.6*labelSize[0][1])),(0,0,0),21)
            # orgimg=cv2ImgAddText(orgimg,result_p,rect_area[0]-height_area,rect_area[1]-height_area-10,(0,255,0),height_area)
               
    print(result_str)
    return orgimg


def process_video(video_path, detect_model, plate_rec_model, device, save_path, fps=None):
    """
    处理单个视频文件或摄像头流，执行车牌检测并保存结果到Excel
    :param video_path: 视频文件路径或摄像头索引（字符串）
    :param detect_model: YOLOv8 检测模型
    :param plate_rec_model: 车牌识别模型
    :param device: 计算设备 (cpu/cuda)
    :param save_path: 结果保存路径
    :param fps: 可选帧率，若为None则从视频中获取
    """
    # 判断是摄像头还是视频文件
    if video_path.isdigit():
        cap = cv2.VideoCapture(int(video_path))
    else:
        cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"无法打开视频源: {video_path}")
        return

    frame_id = 0
    results = []

    # 获取视频帧率（每秒多少帧）
    if fps is None:
        fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print("警告：无法获取有效帧率，设为默认 30 FPS")
        fps = 30.0
    print(f"视频帧率: {fps:.2f} FPS")
    # 计算每秒跳过的帧数（即：处理一帧后，跳过 fps 帧）
    frames_per_second = int(round(fps / 2))  # 每0.5秒处理1帧

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img_ori = copy.deepcopy(frame)
        result_list = det_rec_plate(frame, img_ori, detect_model, plate_rec_model)
        # 获取当前帧的时间（秒）
        now_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        for r in result_list:
            print(now_time, "  ", r)
            # 增加函数判断，如果车牌位置超过4096*1650/2560=2640 说明是对向车道车辆，跳过数据
            rect = r['rect']
            if rect[0] > 2640:  # 过滤对向车道
                continue
            results.append({
                'frame': frame_id,
                'time': now_time,
                'plate_no': r['plate_no'],
                'plate_color': r['plate_color']
            })

        # 跳过接下来的 1 秒（即 skip frames_per_second 帧）
        # 跳过接下来的 ~0.5 秒的帧
        for _ in range(frames_per_second):
            ret = cap.read()[0]
            if not ret:
                break  # 视频结束

        # 更新 frame_id（注意：我们只对处理的帧计数）
        frame_id += frames_per_second + 1  # 当前帧 + 跳过的帧数

        # 可选：实时显示处理的帧
        # cv2.imshow('Processed Frame', draw_result(frame, result_list))
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    # cv2.destroyAllWindows()  # 如果启用显示才需要

    # 保存结果
    video_name = os.path.splitext(os.path.basename(str(video_path)))[0] if not video_path.isdigit() else "camera"
    excel_path = os.path.join(save_path, f'{video_name}_result.xlsx')
    df = pd.DataFrame(results)
    df.to_excel(excel_path, index=False)
    print(f"结果已保存至: {excel_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--detect_model', nargs='+', type=str, default=r'weights/yolov8s.pt', help='model.pt path(s)')
    parser.add_argument('--rec_model', type=str, default=r'weights/plate_rec_color.pth', help='model.pt path(s)')
    parser.add_argument('--image_path', type=str, default=r'imgs', help='source')
    parser.add_argument('--video_path', type=str, default=r'E:\xx过车数据\8月1日(视频)\xxx下午4点-6点主线桥西向东车流\D2_20250801160000_20250801161101.mp4', help='video file or camera index, e.g. 0')
    parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--output', type=str, default='result', help='source')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)]
    opt = parser.parse_args()
    save_path = opt.output

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    detect_model = load_model(opt.detect_model, device)
    plate_rec_model = init_model(device, opt.rec_model, is_color=True)

    # 打印参数量
    total = sum(p.numel() for p in detect_model.parameters())
    total_1 = sum(p.numel() for p in plate_rec_model.parameters())
    print("yolov8 detect params: %.2fM, rec params: %.2fM" % (total/1e6, total_1/1e6))

    detect_model.eval()

    # ========== 视频处理逻辑 ==========
    if opt.video_path:
        video_path = opt.video_path
        if os.path.isdir(video_path):
            # 处理文件夹中所有视频
            video_files = glob.glob(os.path.join(video_path, '*.mp4')) + \
                          glob.glob(os.path.join(video_path, '*.avi')) + \
                          glob.glob(os.path.join(video_path, '*.mov'))
            for vf in video_files:
                print(f"正在处理视频: {vf}")
                process_video(vf, detect_model, plate_rec_model, device, save_path)
        else:
            # 处理单个视频或摄像头
            print(f"正在处理视频源: {video_path}")
            process_video(video_path, detect_model, plate_rec_model, device, save_path)

    # ========== 图片处理逻辑 ==========
    else:
        file_list = []
        allFilePath(opt.image_path, file_list)
        results = []
        for pic_ in file_list:
            img = cv2.imread(pic_)
            if img is None:
                continue
            img_ori = copy.deepcopy(img)
            result_list = det_rec_plate(img, img_ori, detect_model, plate_rec_model)
            for r in result_list:
                results.append({
                    'image': pic_,
                    'plate_no': r['plate_no'],
                    'plate_color': r['plate_color']
                })

        output_filename = f'result_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
        excel_path = os.path.join(save_path, output_filename)
        df = pd.DataFrame(results)
        df.to_excel(excel_path, index=False)
        print(f"图片识别结果已保存至: {excel_path}")