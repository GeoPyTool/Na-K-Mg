import sys
import os
import numpy as np
import pandas as pd
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QMessageBox, QComboBox
)
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt
import matplotlib
matplotlib.use('Agg')

from PySide6.QtWidgets import QWidgetAction
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

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
    eq_up, = ax.plot(x1, y1, 'o-', linewidth=2, markerfacecolor='gray', markeredgecolor='gray', markersize=5, color='gray', label='Equilibrium Upper')
    Na = 100 * np.power(K, 0.37) * np.power(Mg, 0.315)
    S = Na / 1000 + K / 100 + np.sqrt(Mg)
    Na_concentration = Na / 1000 / S
    K_concentration = K / 100 / S
    Mg_concentration = np.sqrt(Mg) / S
    y2 = Na_concentration * np.sin(np.pi/3)
    x2 = Mg_concentration + y2 / np.tan(np.pi/3)
    eq_low, = ax.plot(x2, y2, '^-', linewidth=2, markerfacecolor='gray', markeredgecolor='gray', markersize=5, color='gray', label='Equilibrium Lower')
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
    gig, = ax.plot(x_plot, y, '^-', linewidth=2, markerfacecolor='black', markeredgecolor='black', markersize=7, color='black', label='Giggenbach')
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
    arn, = ax.plot(x_plot, y, 'or-', linewidth=2, markerfacecolor='red', markeredgecolor='red', markersize=7, color='red', label='Arnorsson')
    # Tonani
    Na_K = np.power(10, 883.0 / (t_combined + 273.15) - 0.78)
    T0_Na = K[:len(x)] * Na_K
    S = T0_Na / 1000 + K[:len(x)] / 100 + sqrt_Mg
    T0_Na_concentration = T0_Na / 1000 / S
    T0_Mg_concentration = sqrt_Mg / S
    y = T0_Na_concentration * np.sin(np.pi/3)
    x_plot = T0_Mg_concentration + y / np.tan(np.pi/3)
    ton, = ax.plot(x_plot, y, '*-', linewidth=2, markerfacecolor='green', markeredgecolor='green', markersize=7, color='green', label='Tonani')
    # Fournier
    Na_K = np.power(10, 1217.0 / (t_combined + 273.15) - 1.483)
    Fo_Na = K[:len(x)] * Na_K
    S = Fo_Na / 1000 + K[:len(x)] / 100 + sqrt_Mg
    Fo_Na_concentration = Fo_Na / 1000 / S
    Fo_Mg_concentration = sqrt_Mg / S
    y = Fo_Na_concentration * np.sin(np.pi/3)
    x_plot = Fo_Mg_concentration + y / np.tan(np.pi/3)
    fou, = ax.plot(x_plot, y, '>-', linewidth=2, markerfacecolor='blue', markeredgecolor='blue', markersize=7, color='blue', label='Fournier')
    # Truesdell
    Na_K = np.power(10, 885.6 / (t_combined + 273.15) - 0.8573)
    Tr_Na = K[:len(x)] * Na_K
    S = Tr_Na / 1000 + K[:len(x)] / 100 + sqrt_Mg
    Tr_Na_concentration = Tr_Na / 1000 / S
    Tr_Mg_concentration = sqrt_Mg / S
    y = Tr_Na_concentration * np.sin(np.pi/3)
    x_plot = Tr_Mg_concentration + y / np.tan(np.pi/3)
    tru, = ax.plot(x_plot, y, '<-', linewidth=2, markerfacecolor='cyan', markeredgecolor='cyan', markersize=7, color='cyan', label='Truesdell')
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
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'tab:green', 'tab:blue', 'tab:orange', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive']
    handles = []
    labels = []
    for i in range(len(x)):
        marker = markers[i % len(markers)]
        color = colors[i % len(colors)]
        station_name = data.iloc[i]['台站编号'] if '台站编号' in data.columns else f'Sample {i+1}'
        handle = ax.scatter(x.iloc[i], y.iloc[i], marker=marker, c=color, s=80, label=station_name, edgecolors='black', linewidth=0.5)
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
        # self.ax.set_title('Triangular diagram for Na-K-Mg', fontsize=12, pad=20)
        geo_legend = self.ax.legend(
            [eq_up, eq_low, gig, arn, ton, fou, tru],
            ['Equilibrium Upper', 'Equilibrium Lower', 'Giggenbach', 'Arnorsson', 'Tonani', 'Fournier', 'Truesdell'],
            title='Geothermometer',
            loc='upper left',
            bbox_to_anchor=(0, 1),
            fontsize=9, title_fontsize=9, frameon=True
        )
        sample_legend = self.ax.legend(
            handles=handles, labels=labels, title='Water Samples',
            loc='center left', bbox_to_anchor=(1.0, 0.5),
            fontsize=9, title_fontsize=9, frameon=True
        )
        self.ax.add_artist(geo_legend)
        self.fig.tight_layout()
        # FigureCanvas自适应显示
        if self.canvas:
            self.layout.removeWidget(self.canvas)
            self.canvas.setParent(None)
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        self.canvas = FigureCanvas(self.fig)
        self.layout.addWidget(self.canvas)

    def save_plot(self, filepath, fmt):
        if self.fig:
            self.fig.savefig(filepath, format=fmt)
            return True
        return False

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Na-K-Mg Triangle Plotter")
        self.resize(900, 700)
        self.plot_widget = TrianglePlotWidget()
        self.setCentralWidget(self.plot_widget)
        self.data = None

        # 菜单栏一级操作
        menubar = self.menuBar()
        # 加载文件
        load_action = menubar.addAction("Load File and Plot")
        load_action.triggered.connect(self.load_file)

        # Save format selection
        self.format_combo = QComboBox()
        # 只保留四种格式
        formats = ["png", "jpg", "svg", "pdf"]
        self.format_combo.addItems(formats)
        format_action = QWidgetAction(self)
        format_action.setDefaultWidget(self.format_combo)
        menubar.addAction(format_action)
        # Save
        save_action = menubar.addAction("Save Plot")
        save_action.triggered.connect(self.save_plot)

        # Export classification CSV
        export_action = menubar.addAction("Export Classification CSV")
        export_action.triggered.connect(self.export_classification_csv)

    def classify_sample(self, row):
        # Returns dict of classification for each geothermometer
        result = {}
        Na = row['Na']
        K = row['K']
        Mg = row['Mg']
        # Common calculation
        S = Na / 1000 + K / 100 + np.sqrt(Mg)
        Na_c = Na / 1000 / S
        K_c = K / 100 / S
        Mg_c = np.sqrt(Mg) / S
        y = Na_c * np.sin(np.pi/3)
        x = Mg_c + y / np.tan(np.pi/3)
        # Giggenbach
        if x < 0.33:
            result['Giggenbach'] = 'Mg-dominated'
        elif x < 0.55:
            result['Giggenbach'] = 'Partial Equilibrium'
        else:
            result['Giggenbach'] = 'Equilibrium'
        # Arnorsson
        if x < 0.33:
            result['Arnorsson'] = 'Mg-dominated'
        elif x < 0.55:
            result['Arnorsson'] = 'Partial Equilibrium'
        else:
            result['Arnorsson'] = 'Equilibrium'
        # Tonani
        if x < 0.33:
            result['Tonani'] = 'Mg-dominated'
        elif x < 0.55:
            result['Tonani'] = 'Partial Equilibrium'
        else:
            result['Tonani'] = 'Equilibrium'
        # Fournier
        if x < 0.33:
            result['Fournier'] = 'Mg-dominated'
        elif x < 0.55:
            result['Fournier'] = 'Partial Equilibrium'
        else:
            result['Fournier'] = 'Equilibrium'
        # Truesdell
        if x < 0.33:
            result['Truesdell'] = 'Mg-dominated'
        elif x < 0.55:
            result['Truesdell'] = 'Partial Equilibrium'
        else:
            result['Truesdell'] = 'Equilibrium'
        return result

    def export_classification_csv(self):
        if self.data is None:
            QMessageBox.warning(self, "No Data", "Please load a CSV or Excel file first.")
            return
        df = self.data.copy()
        # For each row, classify
        results = df.apply(self.classify_sample, axis=1)
        for key in ['Giggenbach', 'Arnorsson', 'Tonani', 'Fournier', 'Truesdell']:
            df[key + ' Type'] = results.apply(lambda x: x[key])
        file_path, _ = QFileDialog.getSaveFileName(self, "Export Classification CSV", "", "CSV Files (*.csv)")
        if file_path:
            try:
                df.to_csv(file_path, index=False)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export CSV: {e}")
    def classify_sample(self, row):
        # Returns dict of classification for each geothermometer
        result = {}
        Na = row['Na']
        K = row['K']
        Mg = row['Mg']
        # Common calculation
        S = Na / 1000 + K / 100 + np.sqrt(Mg)
        Na_c = Na / 1000 / S
        K_c = K / 100 / S
        Mg_c = np.sqrt(Mg) / S
        y = Na_c * np.sin(np.pi/3)
        x = Mg_c + y / np.tan(np.pi/3)
        # Giggenbach
        if x < 0.33:
            result['Giggenbach'] = 'Mg-dominated'
        elif x < 0.55:
            result['Giggenbach'] = 'Partial Equilibrium'
        else:
            result['Giggenbach'] = 'Equilibrium'
        # Arnorsson
        if x < 0.33:
            result['Arnorsson'] = 'Mg-dominated'
        elif x < 0.55:
            result['Arnorsson'] = 'Partial Equilibrium'
        else:
            result['Arnorsson'] = 'Equilibrium'
        # Tonani
        if x < 0.33:
            result['Tonani'] = 'Mg-dominated'
        elif x < 0.55:
            result['Tonani'] = 'Partial Equilibrium'
        else:
            result['Tonani'] = 'Equilibrium'
        # Fournier
        if x < 0.33:
            result['Fournier'] = 'Mg-dominated'
        elif x < 0.55:
            result['Fournier'] = 'Partial Equilibrium'
        else:
            result['Fournier'] = 'Equilibrium'
        # Truesdell
        if x < 0.33:
            result['Truesdell'] = 'Mg-dominated'
        elif x < 0.55:
            result['Truesdell'] = 'Partial Equilibrium'
        else:
            result['Truesdell'] = 'Equilibrium'
        return result

    def export_classification_csv(self):
        if self.data is None:
            QMessageBox.warning(self, "No Data", "Please load a CSV or Excel file first.")
            return
        df = self.data.copy()
        # For each row, classify
        results = df.apply(self.classify_sample, axis=1)
        for key in ['Giggenbach', 'Arnorsson', 'Tonani', 'Fournier', 'Truesdell']:
            df[key + ' Type'] = results.apply(lambda x: x[key])
        file_path, _ = QFileDialog.getSaveFileName(self, "Export Classification CSV", "", "CSV Files (*.csv)")
        if file_path:
            try:
                df.to_csv(file_path, index=False)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export CSV: {e}")

    def load_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select CSV or Excel file", "", "CSV Files (*.csv);;Excel Files (*.xlsx *.xls)")
        if file_path:
            try:
                if file_path.endswith('.csv'):
                    data = pd.read_csv(file_path)
                else:
                    data = pd.read_excel(file_path)
                self.data = data
                self.plot_data()  # Auto plot after loading
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load file: {e}")

    def plot_data(self):
        if self.data is not None:
            try:
                self.plot_widget.plot(self.data)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Plotting failed: {e}")
        else:
            QMessageBox.warning(self, "No Data", "Please load a CSV or Excel file first.")
    def save_plot(self):
        if self.plot_widget.fig:
            fmt = self.format_combo.currentText()
            formats = ["svg", "png", "pdf"]
            filters = ";;".join([f"{f.upper()} Files (*.{f})" for f in formats])
            file_path, _ = QFileDialog.getSaveFileName(self, f"Save as {fmt}", "", filters)

            if file_path:

                try:
                    self.plot_widget.fig.savefig(file_path, bbox_inches='tight', dpi=300)
                    QMessageBox.information(self, "Success", f"Plot saved as {file_path}")
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to save plot: {e}")
        else:
            QMessageBox.warning(self, "No Plot", "Please plot the diagram first.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())