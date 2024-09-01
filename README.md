# EsFFT
When using the classic FFT method to detect blurry images, clear images with sparse textures are often classified as blurry, while blurry images with rich textures are classified as clear.

To address this difficulty, we propose an improved version of the ESFFT fuzzy detection method, with the core idea of only considering the frequency response intensity near the texture.

## The detection results of the traditional FFT method are as follows:

### Clear images with sparse textures

<img width="500" alt="9ad8ac80594f5949070337a9cd69661" src="https://github.com/user-attachments/assets/ffd8d1be-cb10-4f75-90ec-07d7b09bdabb">

### Blurred images with rich textures

<img width="500" alt="ecab2bb658c5882389ca529d69295dc" src="https://github.com/user-attachments/assets/772bf0ff-ac29-4a18-8fe7-4bb4fed9dab6">

## The detection results of the ESFFT method are as follows:

### Clear images with sparse textures and it effective response area

<img width="500" alt="e4622134ff55b7838c54a70a92dd23f" src="https://github.com/user-attachments/assets/aaadc793-9657-480e-9c06-73be067a343c">
<img width="500" alt="a4ba840476af46a659689994b7c0134" src="https://github.com/user-attachments/assets/f3565001-0543-4c6b-a195-2bd578c9d582">


### Blurred images with rich textures and it effective response area

<img width="500" alt="5f9385bccd0feebef4b02d34ea1c17c" src="https://github.com/user-attachments/assets/82d0f6d4-0d3d-433d-baff-485d897327c2">
<img width="500" alt="237368f9bed2913843c674327774153" src="https://github.com/user-attachments/assets/3c6f6f5f-2187-48b1-ac18-8ff5fbae6c4a">


## Run Demo

