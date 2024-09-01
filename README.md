# EsFFT
When using the classic FFT method to detect blurry images, clear images with sparse textures are often classified as blurry, while blurry images with rich textures are classified as clear.

To address this difficulty, we propose an improved version of the ESFFT fuzzy detection method, with the core idea of only considering the frequency response intensity near the texture.

The detection results of the traditional FFT method are as follows:

Clear images with sparse textures

<img width="500" alt="9ad8ac80594f5949070337a9cd69661" src="https://github.com/user-attachments/assets/ffd8d1be-cb10-4f75-90ec-07d7b09bdabb">

Blurred images with rich textures

<img width="500" alt="ecab2bb658c5882389ca529d69295dc" src="https://github.com/user-attachments/assets/772bf0ff-ac29-4a18-8fe7-4bb4fed9dab6">

