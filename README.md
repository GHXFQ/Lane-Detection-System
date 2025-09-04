# 车道线检测系统 (Lane Detection System)

基于YOLOv8的车道线分割检测系统，支持图像和视频的车道线检测，并提供Web界面进行交互式处理。

## 🚀 项目特性

- **深度学习模型**: 基于YOLOv8分割模型进行车道线检测
- **多媒体支持**: 支持图像(.jpg, .png, .jpeg)和视频(.mp4, .avi, .mov)处理
- **实时检测**: 支持摄像头实时车道线检测
- **Web界面**: 提供友好的Web界面进行文件上传和结果展示
- **批量处理**: 支持批量图像处理和掩码生成
- **高性能**: 支持GPU加速推理

## 📁 项目结构

```
17011727源码/
├── data/                    # 数据集目录
│   ├── dataset_A/          # 数据集A (测试数据)
│   ├── dataset_B/          # 数据集B
│   └── split/              # 训练数据分割
│       ├── images/         # 图像数据
│       └── labels/         # 标签数据
├── rear/                   # 后端Flask应用
│   ├── app.py             # Flask主应用
│   ├── dist/              # 前端构建文件
│   ├── uploads/           # 上传文件目录
│   └── processed/         # 处理结果目录
├── result/                 # 检测结果输出
├── runs/                   # 训练运行记录
├── ultralytics/           # YOLOv8框架
├── weights/               # 预训练模型权重
├── train.py              # 模型训练脚本
├── detectvedio.py        # 视频检测脚本
├── maskmask.py           # 批量掩码生成脚本
├── myseg.yaml            # 数据集配置文件
├── yolov8-seg.yaml       # 模型配置文件
└── requirements.txt      # 依赖包列表
```

## 🛠️ 环境配置

### 系统要求
- Python 3.8+
- CUDA 11.0+ (可选，用于GPU加速)
- 8GB+ RAM
- 2GB+ 可用磁盘空间

### 安装依赖

```bash
# 安装Python依赖
pip install -r requirements.txt
```

### 依赖包说明
```
Flask==3.0.3              # Web框架
flask-cors==4.0.1          # 跨域支持
numpy==1.26.4              # 数值计算
opencv-python-headless==4.9.0.80  # 计算机视觉
Pillow==10.3.0             # 图像处理
torch==2.3.0               # PyTorch深度学习框架
ultralytics==8.2.38       # YOLOv8框架
```

## 🚀 快速开始

### 1. 模型训练

```bash
# 使用预配置参数训练模型
python train.py
```

训练配置：
- 预训练模型: `yolov8n-seg.pt`
- 训练轮数: 100 epochs
- 图像尺寸: 1280x1280
- 数据集: 根据 `myseg.yaml` 配置

### 2. 视频检测

```bash
# 检测视频中的车道线
python detectvedio.py
```

功能特性：
- 实时FPS显示
- GPU加速推理 (CUDA)
- 置信度阈值: 0.6
- 输出处理后的视频文件

### 3. 批量图像处理

```bash
# 批量生成车道线掩码
python maskmask.py
```

处理流程：
- 读取测试图像目录
- 生成二值化掩码
- 保存检测结果和掩码图像

### 4. Web应用启动

```bash
# 启动Flask后端服务
cd rear
python app.py
```

访问 `http://localhost:5001` 使用Web界面进行文件上传和处理。

## 📊 数据集格式

### 目录结构
```
data/
├── dataset_A/
│   └── test/
│       ├── img/           # 测试图像
│       └── label/         # 对应标签
└── split/
    ├── images/
    │   ├── train/         # 训练图像
    │   ├── val/           # 验证图像
    │   └── test/          # 测试图像
    └── labels/
        ├── train/         # 训练标签
        ├── val/           # 验证标签
        └── test/          # 测试标签
```

### 标签格式
- 使用YOLO分割格式
- 类别: `0: sloid_line` (实线)
- 标注格式: 多边形坐标点

## 🎯 模型配置

### 数据集配置 (myseg.yaml)
```yaml
path: data/dataset_A       # 数据根目录
train: images/train        # 训练图像路径
val: images/val           # 验证图像路径
test: images/test         # 测试图像路径

names:
  0: sloid_line           # 车道线类别
```

### 模型架构 (yolov8-seg.yaml)
- 基础模型: YOLOv8n-seg
- 类别数量: 1 (车道线)
- 输入尺寸: 可变 (推荐1280x1280)
- 输出: 分割掩码 + 边界框

## 🔧 API接口

### 文件上传接口
```
POST /api/upload
Content-Type: multipart/form-data

参数:
- file: 图像或视频文件

响应:
{
  "code": 20000,
  "status": 200,
  "message": "上传成功",
  "data": {
    "result_image": "/processed/filename.jpg"  // 或 result_video
  }
}
```

### 结果获取接口
```
GET /processed/<filename>
```

## 📈 性能指标

### 模型性能
- **推理速度**: ~30-50 FPS (GPU)
- **模型大小**: ~6MB (YOLOv8n)
- **精度**: 根据数据集质量而定
- **支持分辨率**: 320x320 - 1280x1280

### 系统性能
- **内存占用**: ~2-4GB (含模型)
- **GPU显存**: ~1-2GB (推理时)
- **处理延迟**: <100ms (单张图像)

## 🛡️ 注意事项

1. **模型路径**: 确保训练后的模型权重路径正确
2. **GPU支持**: 安装对应CUDA版本的PyTorch
3. **内存管理**: 处理大视频时注意内存使用
4. **文件权限**: 确保输出目录有写入权限
5. **依赖版本**: 严格按照requirements.txt安装依赖

## 🔄 开发流程

### 训练新模型
1. 准备标注数据集
2. 更新 `myseg.yaml` 配置
3. 运行 `train.py` 开始训练
4. 评估模型性能
5. 更新推理脚本中的模型路径

### 部署到生产环境
1. 使用 `gunicorn` 替代Flask开发服务器
2. 配置Nginx反向代理
3. 设置SSL证书
4. 配置日志和监控

## 📝 更新日志

- **v1.0.0**: 初始版本发布
  - 基础车道线检测功能
  - Web界面支持
  - 批量处理能力

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目基于 MIT 许可证开源 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - 提供强大的目标检测框架
- [OpenCV](https://opencv.org/) - 计算机视觉处理库
- [Flask](https://flask.palletsprojects.com/) - 轻量级Web框架

---

⭐ 如果这个项目对你有帮助，请给它一个星标！