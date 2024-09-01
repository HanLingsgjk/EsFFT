import cv2
import numpy as np
import matplotlib.pyplot as plt

# 模糊检测函数
def ESFFT_detect_blur(image, size, thresh):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (h, w) = image.shape
    (cX, cY) = (int(w / 2.0), int(h / 2.0))
    fft = np.fft.fft2(image)  # 使用NumPy的内置算法计算FFT
    fft_shift = np.fft.fftshift(fft)  # 将结果的零频率分量(直流分量)移到中心以便于分析
    fft_shift[cY - size:cY + size, cX - size:cX + size] = 0  # 设置FFT移动为0(即去除低频率)
    fft_shift = np.fft.ifftshift(fft_shift)  # 应用反向位移将DC组件放回左上角
    recon = np.fft.ifft2(fft_shift)  # 应用逆FFT
    magnitude_spectrum = 20 * np.log(np.abs(recon))  # 将中心DC值归零之后，再次计算重建图像的幅度值
    image = image/255.0
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=7)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=7)
    # 只统计梯度明显区域的方向
    val = np.sqrt(sobelx * sobelx + sobely * sobely)
    masku = (val > 24).astype(np.float64)
    maskus = cv2.blur(masku, (25, 25))

    mean = np.mean(magnitude_spectrum[maskus > 0.1])  # 计算幅度值的平均值
    print(mean)
    blurry = mean <= thresh
    return mean, blurry


# 模糊检测函数
def FFT_detect_blur(image, size, thresh):

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        (h, w) = image.shape
        (cX, cY) = (int(w / 2.0), int(h / 2.0))
        fft = np.fft.fft2(image)  # 使用NumPy的内置算法计算FFT
        fft_shift = np.fft.fftshift(fft)  # 将结果的零频率分量(直流分量)移到中心以便于分析
        fft_shift[cY - size:cY + size, cX - size:cX + size] = 0  # 设置FFT移动为0(即去除低频率)
        fft_shift = np.fft.ifftshift(fft_shift)  # 应用反向位移将DC组件放回左上角
        recon = np.fft.ifft2(fft_shift)  # 应用逆FFT
        magnitude_spectrum = 20 * np.log(np.abs(recon))  # 将中心DC值归零之后，再次计算重建图像的幅度值
        mean = np.mean(magnitude_spectrum)  # 计算幅度值的平均值
        print(mean)
        blurry = mean <= thresh
        return mean, blurry


def main():
    image = cv2.imread('/home/lh/all_datasets/blur3.jpg')
    image = cv2.resize(image, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)

    (mean, blurry) = FFT_detect_blur(image, 70, 30)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    color = (0, 0, 255) if blurry else (0, 255, 0)
    text = "Blurry ({:.4f})" if blurry else "Not Blurry ({:.4f})"
    text = text.format(mean)
    cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 5)

    plt.imshow(image)
    plt.show()

if __name__ == '__main__':
    main()

