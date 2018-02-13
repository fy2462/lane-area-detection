# lane-area-detection

## 简介
本项目通过分析检测视频中每一帧的车道线，检测车道临近区域并做相应标记，再转换为视频输出。

运行步骤：

1. 标定相机,获得相机内参
2. 矫正图像曲度
3. 车道线二值提取
4. 转换鹰眼视图
5. 检测左右车道，并拟合出车道曲线
6. 更新车道区域，并转换回原始视图

## 详细介绍

### 标定相机

**calibrate_camera.py**:
使用OpenCV相机标定函数, 从不同的角度和距离对标定棋盘进行标定。calibrate_cam()函数会寻找棋盘图像的内角.
棋盘图例如下：

### 矫正图像

计算得到的相机内参可以用于矫正视频图像，修正由于相机广角造成的图像畸变。undist()用于对视频帧的图像处理工作。
处理前图像：

处理后图像：

可以观察到，图像边缘部分被修正了，为接下来的视图变换奠定了基础。

### 车道线提取

使用灰度二值进行提取。

threshold_helpers.py 包含了多种opencv的二值提取方法，其中包括：

1. abs_sobel_thresh(): 沿x轴或y轴, 在给定灰度阀值和过滤器(kernel)大小情况下, 进行边缘检测，形成二值图像。
2. hls_thresh(): 使用hls(色相，亮度，饱和度)色谱的饱和度通道作为二进制阀值, 形成二值图像。
3. hsv_thresh(): 使用hsv(色相，饱和度，明度)色谱的明度通道作为二进制阀值，形成二值图像。
4. mag_thresh(): 混合了x轴和y轴的abs_sobel_thresh方案，根据两个方向的二范数决定二值图像。
5. dir_thresh(): 使用0 (竖直方向)到pi/2 (水平方向)做为阀值的abs_sobel_thresh方案。
6. combo_thresh(): 组合函数，用于组合上诉几种二值算法。

通过组合观察，最终发现fls和hsv的组合或abs_sobel_thresh的x轴、y轴组合为组合中最佳的分类算法。

调试图像如下：


### 鹰眼视图转换

draw_lane.py 包含了鹰眼视图的转换方法。

change_perspective():