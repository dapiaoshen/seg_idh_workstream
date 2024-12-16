# seg_idh_workstream

基于多任务的胶质瘤分割和IDH分型工作流

工作流介绍：

在医学影像诊断中，脑肿瘤（尤其是胶质瘤）的自动分割与IDH基因突变类型预测是两项关键任务。胶质瘤分割能帮助确定肿瘤区域和特征，而IDH基因型预测（IDH突变或野生型）对患者的治疗规划和预后判断至关重要。然而，这两个任务面临的主要挑战是肿瘤在影像上的高度异质性，以及IDH标注数据稀缺、获取成本昂贵。以往研究多采用单任务方法（只做分割或只做基因型预测），缺乏对任务间关联性的利用。实际上，肿瘤的形态学特征（如影像中的增强模式、边缘清晰度）与基因型信息存在潜在关联。因此，我们在本工作流中将这两个任务整合到同一模型中，将胶质瘤分割与IDH基因型预测有机结合，利用混合CNN-Transformer编码器高效提取多模态MRI特征，为脑肿瘤诊断提供更全面和精准的辅助。

框架图：
![image](https://github.com/user-attachments/assets/0846890a-ff31-4a60-813d-321eb05500d5)

硬件开发环境:
处理器：Intel(R) Xeon(R) Gold 6230 CPU @ 2.10GHz
内存：251GB
硬盘：15.7TB HDD
显卡：NVIDIA A100 GPU

软件开发环境：
Nextflow：22.10.0
Python：3.8.0
torch：2.0.1+cu118
torchvision：0.15.2+cu118
torchaudio：2.0.2+cu118
NumPy：1.21.0
nibabel: 5.4.0

测试集描述
本研究将 BraTS2020 数据集用作测试集。BraTS2020（Brain Tumor Segmentation Challenge 2020）是一个用于脑肿瘤多模态 MRI 分割的权威公开数据集，涵盖了多家医疗机构的预手术 MRI 数据。该数据集包含多种 MRI 模态（如 T1、T1ce、T2、FLAIR），其中包含不同等级的胶质瘤（低级别LGG与高级别HGG）。数据集中每例患者均具有专家手动标注的肿瘤区域掩码，标注包含肿瘤内部结构，如坏死/非增强肿瘤核、增强肿瘤区和肿瘤周边水肿区。通过使用该数据集，我们可以在真实且具代表性的临床数据上验证模型的鲁棒性和泛化能力

数据链接：通过百度网盘分享的文件：testdata
链接：https://pan.baidu.com/s/1TaJmURC7BQfV4nXVFFF7jQ?pwd=84ce 
提取码：84ce 

模型权重链接：通过百度网盘分享的文件：checkpoint
链接：https://pan.baidu.com/s/13elUVK8Y3kF9GGFGlAKeVw?pwd=7dje 
提取码：7dje 

环境安装

conda create -n MTTU python=3.9
conda activate MTTU

安装依赖：
pip install torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2
pip install -r requirements.txt

运行流程：
python main.py









