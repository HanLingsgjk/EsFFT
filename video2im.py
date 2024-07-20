#This function extracts clear images from videos
import os
from absl import app
import gin
import cv2
import numpy as np
import torch
import shutil
def delete_folder(folder_path):
        # If the folder exists, delete it
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)

m = torch.nn.MaxPool2d(100, stride=50)
patch_size = 128
len_for_v = 128
faster = 2
# Fuzzy detection function
def detect_blur_fft(image, size, thresh):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (h, w) = image.shape
    (cX, cY) = (int(w / 2.0), int(h / 2.0))
    imageu = image / 255.0

    image = torch.from_numpy(image).cuda()
    fft = torch.fft.fft2(image)  # Calculate FFT using Torch's built-in algorithm
    fft_shift = torch.fft.fftshift(fft)  # Move the zero frequency component (DC component) of the result to the center for analysis purposes
    fft_shift[cY - size:cY + size, cX - size:cX + size] = 0  # Set FFT shift to 0 (i.e. remove low frequencies)
    fft_shift = torch.fft.ifftshift(fft_shift)  # Apply reverse displacement to place the DC component back in the upper left corner
    recon = torch.fft.ifft2(fft_shift)  # Apply inverse FFT
    recon = recon.detach().cpu().numpy()
    magnitude_spectrum = 20 * np.log(np.abs(recon))  # After resetting the central DC value to zero, calculate the amplitude value of the reconstructed image again

    sobelx = cv2.Sobel(imageu,cv2.CV_64F,1,0,ksize=7)
    sobely = cv2.Sobel(imageu,cv2.CV_64F,0,1,ksize=7)
    #只统计梯度明显区域的方向
    val = np.sqrt(sobelx*sobelx+sobely*sobely)

    masku = (val>24).astype(np.float64)
    maskus = cv2.blur(masku, (25, 25))
    mean = np.mean(magnitude_spectrum[maskus>0])  # 计算幅度值的平均值
    print(mean)
    blurry = mean <= thresh
    return mean, blurry

def main(unused_argv):
    imroot ='/home/lh/all_datasets/video_in510/'
    dirictor = os.listdir(imroot)
    deletcount = 0
    for idx in range(2,dirictor.__len__()):
        # TODO Start reading video
        video_path = imroot+dirictor[idx]
        #TODO Create VideoCapture object
        cap = cv2.VideoCapture(video_path)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        Tlen = frames/len_for_v
        fps = cap.get(cv2.CAP_PROP_FPS)

        Tlen = Tlen/faster

        #TODO My idea is to use a slider, fill it up and send it to CUDA for calculation, maxpooling 2D, find the best image block, and then save it

        image_folder = '/home/lh/all_datasets/video_imout/'+str(idx+0 ).zfill(4)+'/images/'
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)

        # 读取视频
        success, image = cap.read()
        count = 0
        countim = 0

        imvalue_list = []
        count_list = []
        image_list = []
        lastdim = -10
        while success:
            success, image = cap.read()
            if count%faster==0:
                if success:

                    image_list.append(image)
                    count_list.append(count)

                if image_list.__len__()>=Tlen:

                    count_list = np.array(count_list)
                    mean_list= []
                    for imu in image_list:
                        (mean, blurry) = detect_blur_fft(imu,120,30)
                        imvalue_list.append(mean)
                        mean_list.append(mean)
                    mean_list = np.array(mean_list)

                    index = np.argmax(mean_list)
                    max_value =  np.max(mean_list)

                    image_save = image_list[index]
                    if max_value>25 and count_list[index]>(lastdim+1):
                        cv2.imwrite(f"{image_folder}{countim:04d}_{max_value:04f}.jpg", image_save)
                        print('shape:', max_value, 'count:', count_list[index])
                        countim = countim + 1
                        lastdim = count_list[index]
                    count_list = []
                    image_list = []
            count += 1
            print(count)

        imvalue_list = np.array(imvalue_list)
        meanuse = imvalue_list.mean()
        #Remove those with too few valid photos directly
        if meanuse<31 or countim<80:
            delete_folder(image_folder)
            deletcount = deletcount -1
        cap.release()
        print(dirictor)


if __name__ == '__main__':
    with gin.config_scope('train'):
        app.run(main)
