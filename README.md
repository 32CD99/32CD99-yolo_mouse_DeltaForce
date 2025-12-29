# 32CD99-yolo_mouse_DeltaForce
DXGI + TensorRT YOLO demo for mouse control ---------Delta target identification-----------Aim assist

功能特性
基于DXGI 桌面复制技术实现桌面画面捕获
基于TensorRT框架实现 GPU 端推理加速
基于 CUDA 完成预处理与后处理计算
可选配基于STM32 FaHID（功能报告协议） 的鼠标控制功能
轻量可视化界面 + 运行时可调参数（帧率、阈值、盲区范围、增益系数 等）

图片/eye.png

运行环境要求
操作系统
推荐：Windows 11 系统
兼容：Windows 10 系统（未完成全量兼容性测试）
显卡
NVIDIA RTX 40 系显卡（当前适配目标：RTX 4060 / 4060Ti）
建议安装英伟达最新版显卡驱动
编译工具
Visual Studio 2022（微软 MSVC 编译器）
CMake 版本 ≥ 3.23
CUDA 工具包（示例版本：12.x 系列）
TensorRT 推理框架（示例版本：10.x 系列）
注意：本代码仓库不包含英伟达相关二进制文件（TensorRT/CUDA 动态链接库），也不包含推理引擎的 .engine 模型文件。

编译构建步骤
一、安装依赖组件
安装 CUDA 工具包
安装 TensorRT（压缩包解压版 / 安装程序版均可）
确保本地环境已配置好 MSVC 编译器与 CMake 工具


放置推理引擎文件至可执行文件同级目录
将 TensorRT 的引擎模型文件放到程序主执行文件同一文件夹下，目录示例：
plaintextdd_trt_yolo.exe  程序主执行文件
416_fp16.engine  TensorRT推理引擎文件


3. 操作快捷键（示例）
鼠标右键 ：启用 / 挂载功能（功能实现后生效）
F6 键 ：切换推理帧率预设值（60/90/120/144 帧）（功能启用后生效）














安装 CUDA 工具包
安装 TensorRT（压缩包解压版 / 安装程序版均可）
确保本地环境已配置好 MSVC 编译器与 CMake 工具
