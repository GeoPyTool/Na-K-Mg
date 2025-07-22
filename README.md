# Na-K-Mg Triangle Plotter

本项目为一个基于 PySide6 和 Matplotlib 的 Na-K-Mg 三角图可视化工具，适用于地热水样品的地球化学分析。

## 功能简介
- 加载 CSV 或 Excel 格式的水样数据文件
- 自动绘制 Na-K-Mg 三角图，包含地热计曲线（Giggenbach、Arnorsson、Tonani、Fournier、Truesdell）及平衡线
- 支持多种图片格式导出（PNG、JPG、SVG、PDF）
- 支持样品分类结果导出为 CSV 文件
- 图形界面友好，操作简便

## 依赖环境
- Python 3.11+
- PySide6
- matplotlib
- pandas
- numpy

可通过如下命令安装依赖：

```bash
pip install PySide6 matplotlib pandas numpy
```

## 使用方法
1. 运行主程序：

```bash
git clone https://github.com/GeoPyTool/Na-K-Mg-plotter.git
cd Na-K-Mg-plotter
python triangle_gui.py
```

2. 在界面菜单栏选择“Load File and Plot”加载水样数据（支持 CSV、Excel 文件）。
3. 加载后自动绘制三角图。
4. 可通过菜单栏选择图片格式并点击“Save Plot”导出图片。
5. 可点击“Export Classification CSV”导出样品分类结果。

## 数据格式要求
数据文件需包含以下列：
- `Na`：钠离子浓度
- `K`：钾离子浓度
- `Mg`：镁离子浓度
- （可选）`台站编号`：样品编号或名称

## 文件说明
- `triangle_gui.py`：主程序及界面逻辑
- `water_sample.csv`、`complete_water_sample.csv`：示例数据文件
- `na_k_mg_triangle.png`、`result.svg` 等：示例输出图片

## 参考文献

- 刘阳,张华美,李冬雅.Na-K-Mg三角图的Matlab实现——以海南岛温泉(深井)数据分析为例[J].地震地磁观测与研究, 2022, 43(5):111-119.
- Giggenbach, W.F. (1988). Geothermal solute equilibria. Derivation of Na-K-Mg-Ca geoindicators. Geochimica et Cosmochimica Acta, 52(12), 2749-2765.
- Arnórsson, S., et al. (1983). The chemistry of geothermal waters in Iceland. III. Chemical geothermometry in geothermal investigations. Geochimica et Cosmochimica Acta, 47(3), 567-577.

## 许可协议
本项目仅供学术交流与学习使用。
