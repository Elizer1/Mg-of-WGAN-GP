import sys
import os
import tkinter as tk
from tkinter import messagebox
import joblib
import pandas as pd

def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

model_YS = joblib.load(resource_path("LGBM_YS5.21.pkl"))
model_ductility = joblib.load(resource_path("LGBM_Ductility5.211.pkl"))
model_UTS = joblib.load(resource_path("CatBoost_UTS3.pkl"))

def predict_YS(alloy_data):
    X_YS = preprocess_data_YS(alloy_data)
    y_pred_YS = model_YS.predict(X_YS)
    return y_pred_YS[0]

def predict_ductility(alloy_data):
    X_ductility = preprocess_data_ductility(alloy_data)
    y_pred_ductility = model_ductility.predict(X_ductility)
    return y_pred_ductility[0]

def predict_UTS(alloy_data):
    X_UTS = preprocess_data_UTS(alloy_data)
    y_pred_UTS = model_UTS.predict(X_UTS)
    return y_pred_UTS[0]

def preprocess_data_YS(df):
    features_columns_YS = [
        'Coded', 'Mg', 'Nd', 'Ce', 'La', 'Zn', 'Sn', 'Al', 'Ca', 'Zr', 'Ag',
        'Ho', 'Mn', 'Y', 'Gd', 'Cu', 'Si', 'Li', 'Yb', 'Th', 'Sb', 'Pr', 'Ga',
        'Fe', 'Ni', 'Sc', 'Tb', 'Dy', 'Er', 'Sr', 'Bi'
    ]
    X_YS = df[features_columns_YS].copy()
    return X_YS

def preprocess_data_ductility(df):
    df_code = pd.get_dummies(df['Coded'], prefix='Coded')
    for i in range(1, 7):
        col = f'Coded_{i}'
        if col not in df_code.columns:
            df_code[col] = 0
    X_code = df_code[[f'Coded_{i}' for i in range(1, 7)]].reset_index(drop=True)

    features_columns_ductility = [
        'Mg', 'Nd', 'Ce', 'La', 'Zn', 'Sn', 'Al', 'Ca', 'Zr', 'Ag', 'Ho',
        'Mn', 'Y', 'Gd', 'Cu', 'Si', 'Li', 'Yb', 'Th', 'Sb', 'Pr', 'Ga',
        'Fe', 'Ni', 'Sc', 'Tb', 'Dy', 'Er', 'Sr', 'Bi'
    ]
    X_num = df[features_columns_ductility].astype(float).reset_index(drop=True)
    X = pd.concat([X_num, X_code], axis=1)

    try:
        expected = list(model_ductility.feature_names_in_)
    except AttributeError:
        expected = list(X.columns)
    for col in expected:
        if col not in X.columns:
            X[col] = 0
    X = X[expected]
    return X

def preprocess_data_UTS(df):
    features_columns_UTS = [
        'Coded', 'Mg', 'Nd', 'Ce', 'La', 'Zn', 'Sn', 'Al', 'Ca', 'Zr', 'Ag',
        'Ho', 'Mn', 'Y', 'Gd', 'Cu', 'Si', 'Li', 'Yb', 'Th', 'Sb', 'Pr', 'Ga',
        'Fe', 'Ni', 'Sc', 'Tb', 'Dy', 'Er', 'Sr', 'Bi'
    ]
    X_UTS = df[features_columns_UTS].copy()
    return X_UTS

class AlloyPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Mg合金性能预测软件")
        self.root.geometry("425x780")
        self.create_widgets()

    def create_widgets(self):
        main = tk.Frame(self.root)
        main.pack(fill="both", expand=True, padx=8, pady=10)

        # ======= 左侧表格区 =======
        table_frame = tk.LabelFrame(main, text="元素", font=("Arial", 10, "bold"), padx=8, pady=6)
        table_frame.grid(row=0, column=0, sticky="nsw", padx=(0, 8), pady=4)

        tk.Label(table_frame, text="元素", font=("Arial", 10, "bold"), width=6, anchor="center").grid(row=0, column=0)
        tk.Label(table_frame, text="Wt %", font=("Arial", 10, "bold"), width=10, anchor="center").grid(row=0, column=1)

        self.elements = [
            'Mg', 'Nd', 'Ce', 'La', 'Zn', 'Sn', 'Al', 'Ca', 'Zr', 'Ag', 'Ho', 'Mn',
            'Y', 'Gd', 'Cu', 'Si', 'Li', 'Yb', 'Th', 'Sb', 'Pr', 'Ga', 'Fe', 'Ni',
            'Sc', 'Tb', 'Dy', 'Er', 'Sr', 'Bi'
        ]
        self.entries = {}
        for i, element in enumerate(self.elements):
            tk.Label(table_frame, text=element, width=6, anchor="e", font=("Arial", 10)).grid(row=i+1, column=0, sticky="w")
            entry = tk.Entry(table_frame, width=12, font=("Consolas", 10), justify="right", relief="sunken")
            entry.grid(row=i+1, column=1, padx=(2, 0), pady=1)
            self.entries[element] = entry
            if element != 'Mg':
                entry.bind('<KeyRelease>', self.update_mg_value)

        # Reset 按钮
        reset_btn = tk.Button(table_frame, text="Reset", width=10, command=self.reset_entries, font=("Arial", 10))
        reset_btn.grid(row=len(self.elements)+2, column=0, columnspan=2, pady=(7, 0))

        # ======= 右侧预测功能区 =======
        right_frame = tk.LabelFrame(main, text="Mg合金性能预测", font=("Arial", 10, "bold"), padx=12, pady=10)
        right_frame.grid(row=0, column=1, sticky="nsw", pady=4)

        coded_label = tk.Label(right_frame, text="Coded值 (1-6)：", font=("Arial", 10), anchor="w")
        coded_label.grid(row=0, column=0, sticky="w")
        self.coded_entry = tk.Entry(right_frame, font=("Arial", 10), width=14)
        self.coded_entry.grid(row=0, column=1, padx=(5,0), pady=4)

        predict_button = tk.Button(
            right_frame, text="预测", command=self.predict,
            font=("Arial", 12, "bold"), bg="#4CAF50", fg="white", width=14
        )
        predict_button.grid(row=1, column=0, columnspan=2, pady=14)

        # ===== 在右侧添加说明 =====
        explain_text = (
            "Coded值说明(工艺)：\n"
            "1: Sand cast\n"
            "2: HPDC\n"
            "3: Cast + HT\n"
            "4: Extruded\n"
            "5: ECAP\n"
            "6: Wrought"
        )
        explain_label = tk.Label(
            right_frame, text=explain_text,
            font=("Arial", 10), anchor="w", justify="left",
            fg="#333"
        )
        explain_label.grid(row=2, column=0, columnspan=2, sticky="w", pady=(15, 0))

    def reset_entries(self):
        for ele, entry in self.entries.items():
            entry.delete(0, tk.END)
        self.entries['Mg'].insert(0, "100")
        self.coded_entry.delete(0, tk.END)

    def update_mg_value(self, event=None):
        other_total = 0
        for element in self.elements:
            if element == 'Mg':
                continue
            val = self.entries[element].get()
            if val.strip() == '':
                continue
            try:
                other_total += float(val)
            except ValueError:
                continue
        mg_val = 100 - other_total
        self.entries['Mg'].delete(0, tk.END)
        if mg_val < 0:
            self.entries['Mg'].insert(0, "0")
        else:
            self.entries['Mg'].insert(0, f"{mg_val:.4f}")

    def predict(self):
        try:
            alloy_data = {}
            total_percentage = 0
            mg_value = None
            for element in self.elements:
                val = self.entries[element].get()
                if val.strip() == '':
                    val = 0
                alloy_data[element] = float(val)
                total_percentage += alloy_data[element]
                if element == 'Mg':
                    mg_value = float(val)
            if mg_value is None or mg_value < 65:
                raise ValueError("Mg元素的含量必须大于等于65！")
            coded_val = self.coded_entry.get()
            if not coded_val.isdigit() or not (1 <= int(coded_val) <= 6):
                raise ValueError("Coded值必须是1到6之间的整数")
            alloy_data['Coded'] = int(coded_val)
            if abs(total_percentage - 100) > 0.01:
                raise ValueError("合金成分的总和必须为100！")
        except ValueError as e:
            messagebox.showerror("输入错误", f"请输入正确的数字：{e}")
            return

        df = pd.DataFrame([alloy_data])
        try:
            y_pred_YS = predict_YS(df)
            y_pred_ductility = predict_ductility(df)
            y_pred_UTS = predict_UTS(df)
            result = (
                f"屈服强度（YS）预测结果: {y_pred_YS:.2f}\n"
                f"延展性（Ductility）预测结果: {y_pred_ductility:.2f}\n"
                f"拉伸强度（UTS）预测结果: {y_pred_UTS:.2f}"
            )
            messagebox.showinfo("预测结果", result)
        except Exception as e:
            messagebox.showerror("预测失败", f"模型预测出错：{e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = AlloyPredictionApp(root)
    root.mainloop()
