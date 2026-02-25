# 论文展示场景生成器

一个专为学术论文配图设计的仿真场景生成系统，能够创建包含12个YCB物体的结构化堆叠场景。

## 📋 功能特点

### 🎯 场景设计
- **目标物体O_i**: 作为操作目标的底层物体
- **直接风险物体**: 直接压在目标物体上，形成部分遮挡
- **间接风险物体**: 在直接风险物体上形成子结构（3-4个物体）
- **中性物体**: 散布在场景其他位置，增加视觉真实性（6-7个物体）

### 🔧 配置系统
提供3种预设场景配置：

#### 1. Balanced（平衡配置）- 推荐用于论文
- **目标物体**: `004_sugar_box` - 稳定的糖盒
- **直接风险**: `008_pudding_box` - 布丁盒
- **间接风险**: 金枪鱼罐头、柠檬、桃子
- **特点**: 稳定的堆叠结构，视觉效果清晰

#### 2. Challenging（挑战配置）
- **目标物体**: `004_sugar_box`
- **直接风险**: `009_gelatin_box` - 细长不稳定物体
- **间接风险**: 苹果、香蕉、柠檬（易滚动物体）
- **特点**: 不稳定结构，展示复杂操作挑战

#### 3. Realistic（真实配置）
- **目标物体**: `006_mustard_bottle` - 瓶状物体
- **直接风险**: `004_sugar_box` - 大盒子压在瓶子上
- **特点**: 模拟真实杂乱环境

### 📸 相机系统
提供2种相机配置风格：

#### Paper Presentation（论文展示）
- **主相机**: 45度俯视角，1280×960分辨率，展示整体结构
- **侧面相机**: 水平侧视，强调堆叠高度
- **顶部相机**: 鸟瞰图，展示空间布局

#### Detailed Analysis（详细分析）
- **特写相机**: 近距离展示物体细节，1920×1080高清
- **低角度相机**: 仰视角度，增强视觉冲击力

## 🚀 使用方法

### 1. 基本使用

```bash
# 生成默认场景（平衡配置，论文展示风格）
python test_paper_scene.py

# 指定配置和相机风格
python test_paper_scene.py --config challenging --camera detailed_analysis

# 自定义输出目录
python test_paper_scene.py --output ./my_scene_images
```

### 2. 批量生成

```bash
# 生成所有配置组合的图像
python test_paper_scene.py --batch

# 批量生成并创建对比图像（需要ImageMagick）
python test_paper_scene.py --batch --montage
```

### 3. 配置测试

```bash
# 仅测试配置系统，不生成图像
python test_paper_scene.py --test-config
```

### 4. 程序化使用

```python
import gymnasium as gym
from paper_scene import PaperSceneEnv

# 创建环境
env = gym.make("PaperScene-v1", 
               scene_config='balanced',
               camera_style='paper_presentation',
               num_envs=1)

# 初始化场景
obs, _ = env.reset()

# 保存多角度图像
env.unwrapped.save_scene_images("./output_dir")

env.close()
```

## 📁 输出文件结构

```
paper_scene_images_balanced_20231201_143022/
├── main_camera_主视角_45度俯视.png        # 主要展示图像
├── side_camera_侧面视角_展示堆叠高度.png   # 侧面视图
├── top_camera_顶视角_鸟瞰图.png           # 俯视图
└── scene_config.txt                      # 配置信息文档
```

## 🔧 自定义配置

### 创建新的场景配置

编辑 `scene_config.py`：

```python
# 添加新配置
CUSTOM_CONFIG = {
    'target_object': '你的目标物体',
    'direct_risk': '直接风险物体', 
    'indirect_risks': ['间接风险物体1', '间接风险物体2'],
    'neutral_objects': ['中性物体1', '中性物体2', ...],
    'description': '配置描述'
}
```

### 添加新的相机角度

```python
# 在CAMERA_CONFIGS中添加新风格
'custom_style': {
    'new_camera': {
        'eye': [x, y, z],           # 相机位置
        'target': [x, y, z],        # 注视点
        'fov': np.pi / 3,           # 视场角
        'resolution': (w, h),       # 分辨率
        'description': '相机描述'
    }
}
```

## 📊 YCB物体数据库

系统包含12种YCB物体，分类如下：

### 大物体（底层支撑）
- `003_cracker_box`: 饼干盒 [16×21×7cm]
- `004_sugar_box`: 糖盒 [9×17.5×4.4cm] 
- `006_mustard_bottle`: 芥末瓶 [9.5×9.5×17.7cm]

### 中型物体（中间层）
- `008_pudding_box`: 布丁盒 [7.8×10.9×3.2cm]
- `009_gelatin_box`: 明胶盒 [2.8×8.5×11.4cm]
- `010_potted_meat_can`: 罐装肉罐头 [10.1×5.1×5.1cm]

### 小物体（顶层装饰）
- `005_tomato_soup_can`: 番茄汤罐头 [6.5×6.5×10.1cm]
- `007_tuna_fish_can`: 金枪鱼罐头 [8.5×8.5×3.2cm]
- `011_banana`: 香蕉 [18×5.5×5.5cm]
- `013_apple`: 苹果 [9×9×10.5cm]
- `014_lemon`: 柠檬 [5.5×5.5×8cm]
- `015_peach`: 桃子 [7×7×8.5cm]

## 🛠️ 系统要求

### 必需依赖
- Python 3.8+
- ManiSkill2
- SAPIEN
- PyTorch
- NumPy

### 可选依赖
- ImageMagick（用于创建对比图像）
- PIL/Pillow（图像处理）
- Matplotlib（备用可视化）

### 安装方法
```bash
# 安装基础依赖
pip install mani-skill sapien torch numpy pillow

# 安装可选依赖
sudo apt install imagemagick  # Ubuntu/Debian
# 或
brew install imagemagick       # macOS
```

## 📈 性能参数

- **场景稳定化**: 100个物理仿真步骤
- **分辨率**: 1280×960（论文展示）/ 1920×1080（详细分析）
- **生成时间**: 约30-60秒每个场景
- **输出格式**: PNG，RGB图像

## 🎨 论文使用建议

### 图像质量优化
1. 使用 `paper_presentation` 风格获得最佳论文配图效果
2. `balanced` 配置提供最清晰的视觉层次结构
3. 主相机图像适合作为论文主图
4. 侧面图像适合展示堆叠复杂度

### 对比展示
```bash
# 生成不同配置的对比图像
python test_paper_scene.py --batch --montage
```

### 自定义标注
生成的图像保存为PNG格式，可以使用任何图像编辑软件添加标注：
- 目标物体标记
- 风险评估区域
- 操作路径指示
- 文字说明

## 🐛 故障排除

### 常见问题

1. **环境无法创建**
   ```
   解决：检查ManiSkill和SAPIEN是否正确安装
   ```

2. **物体生成失败**  
   ```
   解决：确保YCB资源已下载（asset_download_ids=["ycb"]）
   ```

3. **图像保存失败**
   ```
   解决：检查输出目录权限和磁盘空间
   ```

4. **批量生成中断**
   ```
   解决：检查GPU内存，考虑减少num_envs参数
   ```

### 调试模式
```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 单步调试场景创建
env.unwrapped._create_structured_scene()
```

## 📞 技术支持

如遇到问题，请检查：
1. 系统依赖是否完整安装
2. YCB资源是否正确下载
3. 配置参数是否有效
4. 输出目录是否可写

## 📄 许可证

本项目遵循MIT许可证。适合学术研究和论文使用。

---

**更新日期**: 2024-01-XX  
**版本**: v1.0  
**作者**: RL_Robot Project Team

