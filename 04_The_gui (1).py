import os
import pickle
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# Matplotlib and Seaborn integration for embedded interactive charts
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns


class MLPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Employee Attrition Prediction & Analytics")
        self.root.geometry("1200x850")

        self.df = None
        self.model = None
        self.predictions = None
        self.target_col = None

        self.setup_ui()

    def setup_ui(self):
        # --- Top Action Frame ---
        top_frame = ttk.LabelFrame(self.root, text=" Workflow Controls ", padding=10)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        self.btn_load_model = ttk.Button(
            top_frame, text="1. Load Model (.pkl / .joblib)", command=self.load_model
        )
        self.btn_load_model.pack(side=tk.LEFT, padx=5)

        self.btn_upload = ttk.Button(
            top_frame, text="2. Upload Dataset (.csv)", command=self.upload_csv
        )
        self.btn_upload.pack(side=tk.LEFT, padx=5)

        # Target Column / Attribute Selector
        ttk.Label(top_frame, text="Target Attribute:").pack(side=tk.LEFT, padx=(15, 2))
        self.target_var = tk.StringVar(value="-- Auto Detect --")
        self.combo_target = ttk.Combobox(
            top_frame, textvariable=self.target_var, state="readonly", width=18
        )
        self.combo_target.pack(side=tk.LEFT, padx=5)

        self.btn_predict = ttk.Button(
            top_frame,
            text="3. Run Attrition Prediction",
            command=self.run_prediction,
            state=tk.DISABLED,
        )
        self.btn_predict.pack(side=tk.LEFT, padx=5)

        self.btn_save = ttk.Button(
            top_frame,
            text="4. Export Results (.csv)",
            command=self.save_results,
            state=tk.DISABLED,
        )
        self.btn_save.pack(side=tk.LEFT, padx=5)

        # Restart / Reset Button
        self.btn_reset = ttk.Button(
            top_frame,
            text="🔄 Start Again / Reset",
            command=self.reset_application,
        )
        self.btn_reset.pack(side=tk.RIGHT, padx=5)

        # Status Bar
        self.status_var = tk.StringVar(value="Status: Please load a model and CSV dataset.")
        status_bar = ttk.Label(
            self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, padding=5
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # --- Main Split Notebook ---
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Tab 1: Data View
        self.tab_data = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_data, text=" Data View ")

        table_frame = ttk.Frame(self.tab_data)
        table_frame.pack(fill=tk.BOTH, expand=True)

        scroll_x = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL)
        scroll_y = ttk.Scrollbar(table_frame, orient=tk.VERTICAL)

        self.tree = ttk.Treeview(
            table_frame,
            xscrollcommand=scroll_x.set,
            yscrollcommand=scroll_y.set,
            selectmode="extended",
        )

        scroll_x.config(command=self.tree.xview)
        scroll_y.config(command=self.tree.yview)
        scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.pack(fill=tk.BOTH, expand=True)

        # Tab 2: Analytics & Importance Charts
        self.tab_charts = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_charts, text=" Prediction & Analytics Dashboard ")

        self.chart_frame = ttk.Frame(self.tab_charts)
        self.chart_frame.pack(fill=tk.BOTH, expand=True)
        self.canvas = None

    def reset_application(self):
        """Resets application memory, clears GUI views, and prepares for a new session."""
        if self.df is not None or self.predictions is not None:
            confirm = messagebox.askyesno(
                "Confirm Reset",
                "Are you sure you want to start again?\nUnsaved predictions will be lost.",
            )
            if not confirm:
                return

        self.df = None
        self.predictions = None
        self.model = None

        self.combo_target["values"] = []
        self.target_var.set("-- Auto Detect --")

        self.tree.delete(*self.tree.get_children())
        self.tree["columns"] = []

        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None

        self.btn_predict.config(state=tk.DISABLED)
        self.btn_save.config(state=tk.DISABLED)
        self.notebook.select(self.tab_data)

        self.status_var.set("Status: Application reset! Please load a model and CSV dataset.")
        messagebox.showinfo("Reset Complete", "The application has been reset to start fresh!")

    def load_model(self):
        file_path = filedialog.askopenfilename(
            title="Select Pickled Model File",
            filetypes=[
                ("Supported Models", "*.pkl *.joblib *.pickle"),
                ("All Files", "*.*"),
            ],
        )
        if not file_path:
            return

        try:
            loaded_obj = None
            try:
                import joblib
                loaded_obj = joblib.load(file_path)
            except Exception:
                with open(file_path, "rb") as f:
                    loaded_obj = pickle.load(f)

            if not hasattr(loaded_obj, "predict") and not hasattr(loaded_obj, "predict_proba"):
                raise AttributeError("The loaded file does not contain a valid model with a 'predict' attribute.")

            self.model = loaded_obj
            filename = os.path.basename(file_path)
            self.status_var.set(f"Status: Loaded model [{filename}] successfully.")
            messagebox.showinfo("Success", f"Model successfully loaded:\n{filename}")

            if self.df is not None:
                self.btn_predict.config(state=tk.NORMAL)

        except Exception as e:
            messagebox.showerror("Model Load Error", f"Unable to load model:\n{str(e)}")

    def validate_csv_against_model(self, temp_df):
        """Validates if uploaded CSV matches required attributes of the model."""
        if self.model is None or not hasattr(self.model, "feature_names_in_"):
            return True, []

        expected_features = set(self.model.feature_names_in_)
        preview_encoded = pd.get_dummies(temp_df.head(10), drop_first=False)
        available_features = set(preview_encoded.columns)

        missing_features = []
        for expected in expected_features:
            if expected not in available_features:
                base_col = expected.split("_")[0]
                if base_col not in temp_df.columns:
                    missing_features.append(base_col if base_col in expected else expected)

        missing_features = sorted(list(set(missing_features)))

        if len(missing_features) > (len(expected_features) / 2):
            return False, missing_features

        return True, []

    def upload_csv(self):
        file_path = filedialog.askopenfilename(
            title="Select Input CSV Dataset", filetypes=[("CSV Files", "*.csv")]
        )
        if not file_path:
            return

        try:
            temp_df = pd.read_csv(file_path)

            is_valid, missing_attributes = self.validate_csv_against_model(temp_df)
            if not is_valid:
                missing_str = "\n• " + "\n• ".join(missing_attributes[:10])
                if len(missing_attributes) > 10:
                    missing_str += f"\n...and {len(missing_attributes) - 10} more."

                messagebox.showerror(
                    "Invalid Dataset Rejected",
                    f"Upload Failed! CSV is missing required attributes:\n{missing_str}",
                )
                return

            self.df = temp_df
            self.display_dataframe(self.df)

            cols = ["-- Auto Detect --"] + list(self.df.columns)
            self.combo_target["values"] = cols
            
            attrition_col = [c for c in self.df.columns if 'attrition' in c.lower()]
            if attrition_col:
                self.target_var.set(attrition_col[0])
            else:
                self.target_var.set("-- Auto Detect --")

            filename = os.path.basename(file_path)
            self.status_var.set(f"Status: Loaded dataset [{filename}] with {len(self.df)} rows.")

            if self.model is not None:
                self.btn_predict.config(state=tk.NORMAL)
            self.btn_save.config(state=tk.DISABLED)

        except Exception as e:
            messagebox.showerror("File Error", f"Could not read or process CSV file:\n{str(e)}")

    def display_dataframe(self, df_to_show):
        self.tree.delete(*self.tree.get_children())
        self.tree["columns"] = list(df_to_show.columns)
        self.tree["show"] = "headings"

        for col in df_to_show.columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=140, anchor=tk.CENTER)

        for _, row in df_to_show.head(1000).iterrows():
            self.tree.insert("", tk.END, values=list(row))

    def preprocess_features(self, df_input):
        X = df_input.copy()

        target_selection = self.target_var.get()
        if target_selection in X.columns and target_selection != "-- Auto Detect --":
            X = X.drop(columns=[target_selection])

        # Clean MonthlyIncome to match training preprocessing
        if "MonthlyIncome" in X.columns:
            X["MonthlyIncome"] = (
                X["MonthlyIncome"]
                .astype(str)
                .str.replace("$", "", regex=False)
                .str.replace(",", "", regex=False)
            )
            X["MonthlyIncome"] = pd.to_numeric(X["MonthlyIncome"], errors="coerce")

        X = pd.get_dummies(X, drop_first=False)

        if hasattr(self.model, "feature_names_in_"):
            expected_features = list(self.model.feature_names_in_)

            for col in expected_features:
                if col not in X.columns:
                    X[col] = 0

            X = X[expected_features]
        else:
            X = X.select_dtypes(include=[np.number])
            if X.empty:
                raise ValueError("No valid numerical features found in dataset for prediction.")

        X = X.fillna(X.median(numeric_only=True)).fillna(0)
        return X

    def run_prediction(self):
        if self.df is None:
            messagebox.showwarning("Warning", "Upload a CSV dataset prior to executing predictions.")
            return

        if self.model is None:
            ans = messagebox.askyesno(
                "No Saved Model Loaded",
                "No trained .pkl model file was loaded.\n\nWould you like to auto-train a temporary DEMO model on random labels to test prediction logic and charts?\n\nWARNING: predictions and charts from this demo model are meaningless and for UI testing only.",
            )
            if ans:
                from sklearn.ensemble import RandomForestClassifier

                X_temp = self.preprocess_features(self.df)
                y_temp = np.random.choice([0, 1], size=len(X_temp))
                self.model = RandomForestClassifier(n_estimators=20, random_state=42)
                self.model.fit(X_temp, y_temp)
                self.status_var.set("Status: DEMO MODE — using a randomly-trained placeholder model. Predictions are not real.")
            else:
                return

        try:
            X_processed = self.preprocess_features(self.df)

            if hasattr(self.model, "predict"):
                raw_preds = self.model.predict(X_processed)
            elif hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(X_processed)
                raw_preds = (probs[:, 1] >= 0.5).astype(int)
            else:
                raise AttributeError("The loaded model object lacks a valid 'predict' or 'predict_proba' attribute.")

            friendly_preds = []
            for p in raw_preds:
                if p == 1 or str(p).lower() in ["1", "yes", "true"]:
                    friendly_preds.append("⚠️ High Risk (Likely to Leave)")
                else:
                    friendly_preds.append("✅ Low Risk (Likely to Stay)")

            self.predictions = friendly_preds
            self.df["Attrition_Prediction"] = friendly_preds

            self.display_dataframe(self.df)
            self.render_charts()

            self.btn_save.config(state=tk.NORMAL)
            if not hasattr(self.model, "feature_names_in_") or self.status_var.get().startswith("Status: DEMO MODE"):
                self.status_var.set("Status: DEMO MODE — predictions are from a placeholder model, not real results.")
            else:
                self.status_var.set("Status: Predictions complete!")

            self.notebook.select(self.tab_charts)
            messagebox.showinfo("Success", "Predictions generated successfully!")

        except Exception as e:
            messagebox.showerror("Prediction Error", f"Model failed to predict on dataset:\n{str(e)}")

    def render_charts(self):
        """ Renders a 4-panel dashboard containing model output metrics and dataset analytical distributions """
        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()

        sns.set_theme(style="whitegrid")
        fig = Figure(figsize=(12, 7.5), dpi=95)
        pred_series = pd.Series(self.predictions)

        # -------------------------------------------------------------
        # Plot 1: Attrition Risk Breakdown (Predicted Data)
        # -------------------------------------------------------------
        ax1 = fig.add_subplot(221)
        counts = pred_series.value_counts()
        colors = ["#2ca02c" if "Low Risk" in str(idx) else "#d62728" for idx in counts.index]

        ax1.bar(counts.index.astype(str), counts.values, color=colors)
        ax1.set_title("Attrition Risk Breakdown (Predicted)", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Risk Level")
        ax1.set_ylabel("Employee Count")
        ax1.tick_params(axis="x", rotation=10)

        # -------------------------------------------------------------
        # Plot 2: Top Predicted Attrition Factors
        # -------------------------------------------------------------
        ax2 = fig.add_subplot(222)
        if hasattr(self.model, "feature_importances_") and hasattr(self.model, "feature_names_in_"):
            importances = self.model.feature_importances_
            features = np.array(self.model.feature_names_in_)

            top_indices = np.argsort(importances)[-10:]
            clean_names = [f.replace("_", " ") for f in features[top_indices]]

            ax2.barh(clean_names, importances[top_indices], color="#1f77b4")
            ax2.set_title("Top Predicted Attrition Factors", fontsize=11, fontweight="bold")
            ax2.set_xlabel("Importance Score")
        else:
            ax2.text(
                0.5, 0.5,
                "Feature importance attributes\nnot exposed by this model.",
                ha="center", va="center", fontsize=10,
            )
            ax2.set_title("Top Predicted Attrition Factors", fontsize=11, fontweight="bold")

        # -------------------------------------------------------------
        # Plot 3: Attrition Balance Overview
        # -------------------------------------------------------------
        ax3 = fig.add_subplot(223)
        target_selection = self.target_var.get()

        if self.df is not None and target_selection in self.df.columns and target_selection != "-- Auto Detect --":
            attr_counts = self.df[target_selection].value_counts()
            sns.barplot(x=attr_counts.index.astype(str), y=attr_counts.values, color="#377eb8", ax=ax3)
        else:
            # Fallback to output predictions distribution
            clean_preds = pred_series.str.extract(r'\((.*?)\)')[0].fillna(pred_series)
            attr_counts = clean_preds.value_counts()
            sns.barplot(x=attr_counts.index.astype(str), y=attr_counts.values, color="#377eb8", ax=ax3)

        ax3.set_title("Attrition Balance (Train/Input Set)", fontsize=11, fontweight="bold")
        ax3.set_xlabel("Attrition")
        ax3.set_ylabel("Number of Employees")

        # -------------------------------------------------------------
        # Plot 4: Attrition Breakdown by Categorical Attribute (e.g. Job Role / Marital Status)
        # -------------------------------------------------------------
        ax4 = fig.add_subplot(224)
        cat_col = None

        # Look for typical categorical features in the dataframe
        for candidate in ["JobRole", "MaritalStatus", "Department", "EducationField"]:
            if self.df is not None and candidate in self.df.columns:
                cat_col = candidate
                break

        if cat_col and self.df is not None:
            # Determine target column or use predictions
            hue_col = target_selection if (target_selection in self.df.columns and target_selection != "-- Auto Detect --") else "Attrition_Prediction"
            
            # Plot breakdown
            sns.countplot(data=self.df, x=cat_col, hue=hue_col, palette=["#377eb8", "#ff7f00"], ax=ax4)
            ax4.set_title(f"Attrition by {cat_col}", fontsize=11, fontweight="bold")
            ax4.set_xlabel(cat_col)
            ax4.set_ylabel("Count")
            ax4.tick_params(axis="x", rotation=35)
            ax4.legend(title="Attrition", loc="upper right")
        else:
            # Display a Correlation Heatmap if categorical breakdown columns are missing
            numeric_df = self.df.select_dtypes(include=[np.number]) if self.df is not None else pd.DataFrame()
            if not numeric_df.empty and numeric_df.shape[1] > 1:
                corr = numeric_df.iloc[:, :8].corr()  # Matrix for first 8 numeric features
                sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax4, cbar=True)
                ax4.set_title("Numeric Feature Correlation", fontsize=11, fontweight="bold")
            else:
                ax4.text(0.5, 0.5, "Categorical/Numeric Attributes\nNot Found in Input CSV", ha="center", va="center")
                ax4.set_title("Attribute Distribution Analysis", fontsize=11, fontweight="bold")

        fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def save_results(self):
        if self.df is None or "Attrition_Prediction" not in self.df.columns:
            messagebox.showwarning("Warning", "No active prediction values found to save.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv")],
            title="Save Prediction Data As",
        )

        if file_path:
            try:
                self.df.to_csv(file_path, index=False)
                messagebox.showinfo("Export Complete", f"Data exported successfully to:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Unable to write file:\n{str(e)}")


if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = MLPredictionApp(root)
        root.mainloop()
    except Exception as err:
        print("Error launching desktop app window:", err)
        input("\nPress Enter to close window...")
