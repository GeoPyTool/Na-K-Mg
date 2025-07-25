import sys
import os
import numpy as np
import pandas as pd
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QFileDialog, QMessageBox, QComboBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QTabWidget, 
    QSplitter, QSizePolicy, QMenuBar, QTextEdit
)
from PySide6.QtGui import QPixmap, QIcon, QAction
from PySide6.QtCore import Qt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.font_manager as fm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 翻译字典
translations = {
    "中文": {
        "Na-K-Mg 三角图解工具": "Na-K-Mg 三角图解工具",
        "文件": "文件",
        "加载数据": "加载数据",
        "保存图像": "保存图像",
        "导出分类结果": "导出分类结果",
        "退出": "退出",
        "语言": "语言",
        "中文": "中文",
        "English": "English",
        "K (mg/L)": "K (mg/L)",
        "Na (mg/L)": "Na (mg/L)",
        "Mg (mg/L)": "Mg (mg/L)",
        "分类结果": "分类结果",
        "x值": "x值",
        "y值": "y值",
        "站点": "站点",
        "详细分类结果": "详细分类结果",
        "Giggenbach": "Giggenbach",
        "Arnorsson": "Arnorsson",
        "Tonani": "Tonani",
        "Fournier": "Fournier",
        "Truesdell": "Truesdell",
        "Equilibrium Upper": "平衡上限",
        "Equilibrium Lower": "平衡下限",
        "Geothermometer": "地热温标",
        "Water Samples": "水样",
        "Mg-dominated": "镁主导",
        "Partial Equilibrium": "部分平衡",
        "Equilibrium": "平衡",
        "加载CSV或Excel文件": "加载CSV或Excel文件",
        "保存图像为": "保存图像为",
        "CSV文件": "CSV文件",
        "Excel文件": "Excel文件",
        "请选择CSV或Excel文件": "请选择CSV或Excel文件",
        "图像保存成功": "图像保存成功",
        "数据加载成功": "数据加载成功",
        "请选择保存路径": "请选择保存路径",
        "请先加载数据": "请先加载数据",
        "导出成功": "导出成功",
        "错误": "错误",
        "成功": "成功",
        "警告": "警告"
    },
    "English": {
        "Na-K-Mg 三角图解工具": "Na-K-Mg Triangle Diagram Tool",
        "文件": "File",
        "加载数据": "Load Data",
        "保存图像": "Save Plot",
        "导出分类结果": "Export Classification Results",
        "退出": "Exit",
        "语言": "Language",
        "中文": "Chinese",
        "English": "English",
        "K (mg/L)": "K (mg/L)",
        "Na (mg/L)": "Na (mg/L)",
        "Mg (mg/L)": "Mg (mg/L)",
        "分类结果": "Classification",
        "x值": "x-value",
        "y值": "y-value",
        "站点": "Station",
        "详细分类结果": "Detailed Classification Results",
        "Giggenbach": "Giggenbach",
        "Arnorsson": "Arnorsson",
        "Tonani": "Tonani",
        "Fournier": "Fournier",
        "Truesdell": "Truesdell",
        "Equilibrium Upper": "Equilibrium Upper",
        "Equilibrium Lower": "Equilibrium Lower",
        "Geothermometer": "Geothermometer",
        "Water Samples": "Water Samples",
        "Mg-dominated": "Mg-dominated",
        "Partial Equilibrium": "Partial Equilibrium",
        "Equilibrium": "Equilibrium",
        "加载CSV或Excel文件": "Load CSV or Excel file",
        "保存图像为": "Save plot as",
        "CSV文件": "CSV Files",
        "Excel文件": "Excel Files",
        "请选择CSV或Excel文件": "Please select CSV or Excel file",
        "图像保存成功": "Plot saved successfully",
        "数据加载成功": "Data loaded successfully",
        "请选择保存路径": "Please select save path",
        "请先加载数据": "Please load data first",
        "导出成功": "Export successful",
        "错误": "Error",
        "成功": "Success",
        "警告": "Warning"
    }
}

# 全局语言变量
language = "中文"

def tr(text):
    """翻译函数"""
    return translations.get(language, translations["中文"]).get(text, text)

def draw_triangle():
    fig, ax = plt.subplots(figsize=(8, 8))
    vA = np.array([0, 0])
    vB = np.array([1, 0])
    vC = np.array([0.5, np.sqrt(3)/2])
    
    # Draw triangle edges
    ax.plot([vA[0], vB[0]], [vA[1], vB[1]], 'k-', lw=2)
    ax.plot([vB[0], vC[0]], [vB[1], vC[1]], 'k-', lw=2)
    ax.plot([vC[0], vA[0]], [vC[1], vA[1]], 'k-', lw=2)

    # Internal grid lines (parallel to edges)
    for frac in np.arange(0.1, 1, 0.1):
        # Parallel to AB
        ax.plot([vA[0]*(1-frac)+vC[0]*frac, vB[0]*(1-frac)+vC[0]*frac],
                [vA[1]*(1-frac)+vC[1]*frac, vB[1]*(1-frac)+vC[1]*frac], '--', color='lightgray', lw=0.8)
        # Parallel to BC
        ax.plot([vB[0]*(1-frac)+vA[0]*frac, vC[0]*(1-frac)+vA[0]*frac],
                [vB[1]*(1-frac)+vA[1]*frac, vC[1]*(1-frac)+vA[1]*frac], '--', color='lightgray', lw=0.8)
        # Parallel to CA
        ax.plot([vC[0]*(1-frac)+vB[0]*frac, vA[0]*(1-frac)+vB[0]*frac],
                [vC[1]*(1-frac)+vB[1]*frac, vA[1]*(1-frac)+vB[1]*frac], '--', color='lightgray', lw=0.8)

    # Axis ticks on three sides
    for i, frac in enumerate(np.arange(0, 1.1, 0.1)):
        # Bottom (AB)
        ax.text(frac, -0.01, f'{frac:.1f}', ha='center', va='top', fontsize=7)
        # Right (BC)
        x_right = (1-frac)*vB[0] + frac*vC[0]
        y_right = (1-frac)*vB[1] + frac*vC[1]
        ax.text(x_right+0.01, y_right, f'{frac:.1f}', ha='left', va='center', fontsize=7)
        # Left (CA)
        x_left = (1-frac)*vC[0] + frac*vA[0]
        y_left = (1-frac)*vC[1] + frac*vA[1]
        ax.text(x_left-0.01, y_left, f'{frac:.1f}', ha='right', va='center', fontsize=7)

    # Axis labels
    ax.text(0.5, -0.03, r'$\sqrt{Mg}$', ha='center', va='top', fontsize=12)
    ax.text(0.8, np.sqrt(3)/4, 'Na/1000', ha='left', va='center', fontsize=12)
    ax.text(0.2, np.sqrt(3)/4, 'K/100', ha='right', va='center', fontsize=12)

    # Set aspect ratio strictly to 1:1
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.13, np.sqrt(3)/2 + 0.08)
    ax.axis('off')
    return fig, ax

def draw_equilibrium_lines(ax):
    t = np.arange(80, 361, 20)
    K = np.ones_like(t)
    Lkm = 14.0 - 4410.0 / (t + 273.15)
    K2_Mg = np.power(10, Lkm)
    Mg = K / K2_Mg
    Na = 457 * np.power(K, 0.37) * np.power(Mg, 0.315)
    S = Na / 1000 + K / 100 + np.sqrt(Mg)
    Na_concentration = Na / 1000 / S
    K_concentration = K / 100 / S
    Mg_concentration = np.sqrt(Mg) / S
    y1 = Na_concentration * np.sin(np.pi/3)
    x1 = Mg_concentration + y1 / np.tan(np.pi/3)
    eq_up, = ax.plot(x1, y1, 'o-', linewidth=2, markerfacecolor='gray', markeredgecolor='gray', markersize=5, color='gray', label=tr('Equilibrium Upper'))
    
    Na = 100 * np.power(K, 0.37) * np.power(Mg, 0.315)
    S = Na / 1000 + K / 100 + np.sqrt(Mg)
    Na_concentration = Na / 1000 / S
    K_concentration = K / 100 / S
    Mg_concentration = np.sqrt(Mg) / S
    y2 = Na_concentration * np.sin(np.pi/3)
    x2 = Mg_concentration + y2 / np.tan(np.pi/3)
    eq_low, = ax.plot(x2, y2, '^-', linewidth=2, markerfacecolor='gray', markeredgecolor='gray', markersize=5, color='gray', label=tr('Equilibrium Lower'))
    
    for i in range(len(t)):
        ax.plot([x1[i], x2[i]], [y1[i], y2[i]], '--', color='gray', lw=0.8)
        ax.text(x1[i], y1[i] + 0.02, f"{t[i]:.0f}°C", ha='center', va='bottom', color='gray', fontsize=9)
    
    return eq_up, eq_low

def draw_na_k_curves(ax):
    t = np.arange(75, 351, 25)
    K = np.ones_like(t)
    x1 = 7.35 - 2300.0 / (t + 273.15)
    x2 = 4.03 - 1077.0 / (t + 273.15)
    x = np.concatenate([x2[x2 < 1.25], x1[x1 > 1.25]])
    t_combined = np.concatenate([t[x2 < 1.25], t[x1 > 1.25]])
    K_sqrt_Mg = np.power(10, x)
    sqrt_Mg = K[:len(x)] / K_sqrt_Mg
    
    # Giggenbach
    Na_K = np.power(10, 1390.0 / (t_combined + 273.15) - 1.75)
    Gi_Na = K[:len(x)] * Na_K
    S = Gi_Na / 1000 + K[:len(x)] / 100 + sqrt_Mg
    Gi_Na_concentration = Gi_Na / 1000 / S
    Gi_Mg_concentration = sqrt_Mg / S
    y = Gi_Na_concentration * np.sin(np.pi/3)
    x_plot = Gi_Mg_concentration + y / np.tan(np.pi/3)
    gig, = ax.plot(x_plot, y, '^-', linewidth=2, markerfacecolor='black', markeredgecolor='black', markersize=7, color='black', label=tr('Giggenbach'))
    
    # Arnorsson
    Na_K = np.where(t_combined < 250, 
                    np.power(10, 993.0 / (t_combined + 273.15) - 0.933),
                    np.power(10, 1319.0 / (t_combined + 273.15) - 1.699))
    Ar_Na = K[:len(x)] * Na_K
    S = Ar_Na / 1000 + K[:len(x)] / 100 + sqrt_Mg
    Ar_Na_concentration = Ar_Na / 1000 / S
    Ar_Mg_concentration = sqrt_Mg / S
    y = Ar_Na_concentration * np.sin(np.pi/3)
    x_plot = Ar_Mg_concentration + y / np.tan(np.pi/3)
    arn, = ax.plot(x_plot, y, 'or-', linewidth=2, markerfacecolor='red', markeredgecolor='red', markersize=7, color='red', label=tr('Arnorsson'))
    
    # Tonani
    Na_K = np.power(10, 883.0 / (t_combined + 273.15) - 0.78)
    T0_Na = K[:len(x)] * Na_K
    S = T0_Na / 1000 + K[:len(x)] / 100 + sqrt_Mg
    T0_Na_concentration = T0_Na / 1000 / S
    T0_Mg_concentration = sqrt_Mg / S
    y = T0_Na_concentration * np.sin(np.pi/3)
    x_plot = T0_Mg_concentration + y / np.tan(np.pi/3)
    ton, = ax.plot(x_plot, y, '*-', linewidth=2, markerfacecolor='green', markeredgecolor='green', markersize=7, color='green', label=tr('Tonani'))
    
    # Fournier
    Na_K = np.power(10, 1217.0 / (t_combined + 273.15) - 1.483)
    Fo_Na = K[:len(x)] * Na_K
    S = Fo_Na / 1000 + K[:len(x)] / 100 + sqrt_Mg
    Fo_Na_concentration = Fo_Na / 1000 / S
    Fo_Mg_concentration = sqrt_Mg / S
    y = Fo_Na_concentration * np.sin(np.pi/3)
    x_plot = Fo_Mg_concentration + y / np.tan(np.pi/3)
    fou, = ax.plot(x_plot, y, '>-', linewidth=2, markerfacecolor='blue', markeredgecolor='blue', markersize=7, color='blue', label=tr('Fournier'))
    
    # Truesdell
    Na_K = np.power(10, 885.6 / (t_combined + 273.15) - 0.8573)
    Tr_Na = K[:len(x)] * Na_K
    S = Tr_Na / 1000 + K[:len(x)] / 100 + sqrt_Mg
    Tr_Na_concentration = Tr_Na / 1000 / S
    Tr_Mg_concentration = sqrt_Mg / S
    y = Tr_Na_concentration * np.sin(np.pi/3)
    x_plot = Tr_Mg_concentration + y / np.tan(np.pi/3)
    tru, = ax.plot(x_plot, y, '<-', linewidth=2, markerfacecolor='cyan', markeredgecolor='cyan', markersize=7, color='cyan', label=tr('Truesdell'))
    
    return gig, arn, ton, fou, tru

def plot_samples(ax, data):
    K, Na, Mg = data['K'], data['Na'], data['Mg']
    S = Na / 1000 + K / 100 + np.sqrt(Mg)
    Na_concentration = Na / 1000 / S
    K_concentration = K / 100 / S
    Mg_concentration = np.sqrt(Mg) / S
    y = Na_concentration * np.sin(np.pi/3)
    x = Mg_concentration + y / np.tan(np.pi/3)
    
    markers = ['o', 's', 'D', '^', 'v', '>', '<', 'p', 'h', '*', 'X', 'P', '1', '2', '3', '4', '8', '|', '_']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    handles = []
    labels = []
    
    for i in range(len(x)):
        marker = markers[i % len(markers)]
        color = colors[i % len(colors)]
        station_name = data.iloc[i]['台站编号'] if '台站编号' in data.columns else f'Sample {i+1}'
        handle = ax.scatter(x.iloc[i], y.iloc[i], marker=marker, c=color, s=80, 
                          label=station_name, edgecolors='black', linewidth=0.5)
        handles.append(handle)
        labels.append(station_name)
    
    return handles, labels

class TrianglePlotWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = None
        self.fig = None
        self.ax = None
        self.canvas = None
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

    def plot(self, data):
        self.data = data
        plt.close('all')
        self.fig, self.ax = draw_triangle()
        eq_up, eq_low = draw_equilibrium_lines(self.ax)
        gig, arn, ton, fou, tru = draw_na_k_curves(self.ax)
        handles, labels = plot_samples(self.ax, data)
        
        geo_legend = self.ax.legend(
            [eq_up, eq_low, gig, arn, ton, fou, tru],
            [tr('Equilibrium Upper'), tr('Equilibrium Lower'), tr('Giggenbach'), 
             tr('Arnorsson'), tr('Tonani'), tr('Fournier'), tr('Truesdell')],
            title=tr('Geothermometer'),
            loc='upper left',
            bbox_to_anchor=(0, 1),
            fontsize=9, title_fontsize=9, frameon=True
        )
        
        sample_legend = self.ax.legend(
            handles=handles, labels=labels, title=tr('Water Samples'),
            loc='center left', bbox_to_anchor=(1.0, 0.5),
            fontsize=9, title_fontsize=9, frameon=True
        )
        
        self.ax.add_artist(geo_legend)
        self.fig.tight_layout()
        
        if self.canvas:
            self.layout.removeWidget(self.canvas)
            self.canvas.setParent(None)
        
        self.canvas = FigureCanvas(self.fig)
        self.layout.addWidget(self.canvas)

    def save_plot(self, filepath, fmt):
        if self.fig:
            self.fig.savefig(filepath, format=fmt, bbox_inches='tight', dpi=300)
            return True
        return False

class ResultWindow(QMainWindow):
    def __init__(self, data, parent=None):
        super().__init__(parent)
        self.data = data
        self.setWindowTitle(tr('详细分类结果'))
        self.resize(800, 600)
        self.initUI()

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # 创建标签页
        tab_widget = QTabWidget()
        layout.addWidget(tab_widget)

        # 创建结果标签页
        self.create_result_tab(tab_widget)

    def create_result_tab(self, tab_widget):
        table = QTableWidget()
        table.setColumnCount(7)
        table.setHorizontalHeaderLabels([
            tr('站点'), tr('K (mg/L)'), tr('Na (mg/L)'), tr('Mg (mg/L)'),
            tr('分类结果'), tr('x值'), tr('y值')
        ])

        # 计算分类结果
        results = []
        for idx, row in self.data.iterrows():
            result = self.classify_sample(row)
            results.append(result)

        table.setRowCount(len(self.data))

        for i, (idx, row) in enumerate(self.data.iterrows()):
            table.setItem(i, 0, QTableWidgetItem(str(row.get('台站编号', f'Sample {i+1}'))))
            table.setItem(i, 1, QTableWidgetItem(str(row['K'])))
            table.setItem(i, 2, QTableWidgetItem(str(row['Na'])))
            table.setItem(i, 3, QTableWidgetItem(str(row['Mg'])))
            
            # 获取分类结果
            result = results[i]
            classification = result.get('Giggenbach', 'Unknown')
            table.setItem(i, 4, QTableWidgetItem(tr(classification)))
            
            # 计算x, y值
            Na, K, Mg = row['Na'], row['K'], row['Mg']
            S = Na / 1000 + K / 100 + np.sqrt(Mg)
            Na_concentration = Na / 1000 / S
            K_concentration = K / 100 / S
            Mg_concentration = np.sqrt(Mg) / S
            y = Na_concentration * np.sin(np.pi/3)
            x = Mg_concentration + y / np.tan(np.pi/3)
            
            table.setItem(i, 5, QTableWidgetItem(f"{x:.4f}"))
            table.setItem(i, 6, QTableWidgetItem(f"{y:.4f}"))

        # 设置表格属性
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.setAlternatingRowColors(True)

        tab_widget = QTabWidget()
        tab_widget.addTab(table, tr('分类结果'))
        
        # 添加详细信息标签页
        self.create_detail_tab(tab_widget, results)

        self.setCentralWidget(tab_widget)

    def create_detail_tab(self, tab_widget, results):
        detail_widget = QWidget()
        layout = QVBoxLayout(detail_widget)

        text_edit = QTextEdit()
        text_edit.setReadOnly(True)

        html_content = f"""
        <h3>{tr('详细分类结果')}</h3>
        <table border="1" style="border-collapse: collapse; width: 100%;">
        <tr>
            <th>{tr('站点')}</th>
            <th>{tr('Giggenbach')}</th>
            <th>{tr('Arnorsson')}</th>
            <th>{tr('Tonani')}</th>
            <th>{tr('Fournier')}</th>
            <th>{tr('Truesdell')}</th>
        </tr>
        """

        for i, (idx, row) in enumerate(self.data.iterrows()):
            station = str(row.get('台站编号', f'Sample {i+1}'))
            result = results[i]
            html_content += f"""
            <tr>
                <td>{station}</td>
                <td>{tr(result.get('Giggenbach', 'Unknown'))}</td>
                <td>{tr(result.get('Arnorsson', 'Unknown'))}</td>
                <td>{tr(result.get('Tonani', 'Unknown'))}</td>
                <td>{tr(result.get('Fournier', 'Unknown'))}</td>
                <td>{tr(result.get('Truesdell', 'Unknown'))}</td>
            </tr>
            """

        html_content += "</table>"
        text_edit.setHtml(html_content)
        layout.addWidget(text_edit)

        tab_widget.addTab(detail_widget, tr('详细信息'))

    def classify_sample(self, row):
        """分类单个样本"""
        result = {}
        Na = row['Na']
        K = row['K']
        Mg = row['Mg']
        
        # 通用计算
        S = Na / 1000 + K / 100 + np.sqrt(Mg)
        Na_c = Na / 1000 / S
        K_c = K / 100 / S
        Mg_c = np.sqrt(Mg) / S
        y = Na_c * np.sin(np.pi/3)
        x = Mg_c + y / np.tan(np.pi/3)

        # 分类逻辑
        geothermometers = ['Giggenbach', 'Arnorsson', 'Tonani', 'Fournier', 'Truesdell']
        for geo in geothermometers:
            if x < 0.33:
                result[geo] = 'Mg-dominated'
            elif x < 0.55:
                result[geo] = 'Partial Equilibrium'
            else:
                result[geo] = 'Equilibrium'

        return result

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.data = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle(tr('Na-K-Mg 三角图解工具'))
        self.resize(1000, 700)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        
        # 创建绘图组件
        self.plot_widget = TrianglePlotWidget()
        layout.addWidget(self.plot_widget)

        # 创建菜单栏
        self.create_menu_bar()

    def create_menu_bar(self):
        menubar = self.menuBar()

        # 文件菜单
        file_menu = menubar.addMenu(tr('文件'))

        # 加载数据
        load_action = QAction(tr('加载数据'), self)
        load_action.triggered.connect(self.load_file)
        file_menu.addAction(load_action)

        # 保存图像
        save_action = QAction(tr('保存图像'), self)
        save_action.triggered.connect(self.save_plot)
        file_menu.addAction(save_action)

        # 导出分类结果
        export_action = QAction(tr('导出分类结果'), self)
        export_action.triggered.connect(self.export_classification_csv)
        file_menu.addAction(export_action)

        # 退出
        exit_action = QAction(tr('退出'), self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # 语言菜单
        language_menu = menubar.addMenu(tr('语言'))

        # 中文
        chinese_action = QAction(tr('中文'), self)
        chinese_action.triggered.connect(lambda: self.switch_language('中文'))
        language_menu.addAction(chinese_action)

        # 英文
        english_action = QAction(tr('English'), self)
        english_action.triggered.connect(lambda: self.switch_language('English'))
        language_menu.addAction(english_action)

    def switch_language(self, lang):
        """切换语言"""
        global language
        language = lang
        
        # 清除现有菜单栏
        self.menuBar().clear()
        
        # 重新初始化UI
        self.initUI()
        
        # 如果有数据，重新绘制
        if self.data is not None:
            self.plot_widget.plot(self.data)

    def classify_sample(self, row):
        """分类单个样本"""
        result = {}
        Na = row['Na']
        K = row['K']
        Mg = row['Mg']
        
        # 通用计算
        S = Na / 1000 + K / 100 + np.sqrt(Mg)
        Na_c = Na / 1000 / S
        K_c = K / 100 / S
        Mg_c = np.sqrt(Mg) / S
        y = Na_c * np.sin(np.pi/3)
        x = Mg_c + y / np.tan(np.pi/3)

        # 分类逻辑
        geothermometers = ['Giggenbach', 'Arnorsson', 'Tonani', 'Fournier', 'Truesdell']
        for geo in geothermometers:
            if x < 0.33:
                result[geo] = 'Mg-dominated'
            elif x < 0.55:
                result[geo] = 'Partial Equilibrium'
            else:
                result[geo] = 'Equilibrium'

        return result

    def load_file(self):
        """加载文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, tr('请选择CSV或Excel文件'), "", 
            f"{tr('CSV文件')} (*.csv);;{tr('Excel文件')} (*.xlsx *.xls)"
        )
        
        if file_path:
            try:
                if file_path.endswith('.csv'):
                    data = pd.read_csv(file_path)
                else:
                    data = pd.read_excel(file_path)
                
                self.data = data
                self.plot_widget.plot(data)
                QMessageBox.information(self, tr('成功'), tr('数据加载成功'))
            except Exception as e:
                QMessageBox.critical(self, tr('错误'), f"{tr('错误')}: {e}")

    def save_plot(self):
        """保存图像"""
        if not self.plot_widget.fig:
            QMessageBox.warning(self, tr('警告'), tr('请先加载数据'))
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, tr('保存图像为'), "", 
            "SVG Files (*.svg);;PNG Files (*.png);;PDF Files (*.pdf);;JPG Files (*.jpg)"
        )
        
        if file_path:
            try:
                fmt = file_path.split('.')[-1]
                self.plot_widget.save_plot(file_path, fmt)
                QMessageBox.information(self, tr('成功'), tr('图像保存成功'))
            except Exception as e:
                QMessageBox.critical(self, tr('错误'), f"{tr('错误')}: {e}")

    def export_classification_csv(self):
        """导出分类结果"""
        if self.data is None:
            QMessageBox.warning(self, tr('警告'), tr('请先加载数据'))
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, tr('导出分类结果'), "", f"{tr('CSV文件')} (*.csv)"
        )
        
        if file_path:
            try:
                df = self.data.copy()
                
                # 为每行添加分类结果
                results = df.apply(self.classify_sample, axis=1)
                
                for geo in ['Giggenbach', 'Arnorsson', 'Tonani', 'Fournier', 'Truesdell']:
                    df[f'{geo}_Type'] = results.apply(lambda x: tr(x.get(geo, 'Unknown')))
                
                df.to_csv(file_path, index=False)
                QMessageBox.information(self, tr('成功'), tr('导出成功'))
                
                # 显示详细结果窗口
                self.show_detailed_results()
                
            except Exception as e:
                QMessageBox.critical(self, tr('错误'), f"{tr('错误')}: {e}")

    def show_detailed_results(self):
        """显示详细结果窗口"""
        if self.data is not None:
            self.result_window = ResultWindow(self.data)
            self.result_window.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
