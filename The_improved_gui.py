def render_charts(self):
        """ Renders a 4-panel dashboard using pure Matplotlib (No Seaborn required) """
        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()

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
        ax1.grid(True, linestyle="--", alpha=0.5)

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
        ax2.grid(True, linestyle="--", alpha=0.5)

        # -------------------------------------------------------------
        # Plot 3: Attrition Balance Overview
        # -------------------------------------------------------------
        ax3 = fig.add_subplot(223)
        target_selection = self.target_var.get()

        if self.df is not None and target_selection in self.df.columns and target_selection != "-- Auto Detect --":
            attr_counts = self.df[target_selection].value_counts()
            ax3.bar(attr_counts.index.astype(str), attr_counts.values, color="#377eb8")
        else:
            clean_preds = pred_series.str.extract(r'\((.*?)\)')[0].fillna(pred_series)
            attr_counts = clean_preds.value_counts()
            ax3.bar(attr_counts.index.astype(str), attr_counts.values, color="#377eb8")

        ax3.set_title("Attrition Balance (Train/Input Set)", fontsize=11, fontweight="bold")
        ax3.set_xlabel("Attrition")
        ax3.set_ylabel("Number of Employees")
        ax3.grid(True, linestyle="--", alpha=0.5)

        # -------------------------------------------------------------
        # Plot 4: Attrition Breakdown by Categorical Attribute
        # -------------------------------------------------------------
        ax4 = fig.add_subplot(224)
        cat_col = None

        for candidate in ["JobRole", "MaritalStatus", "Department", "EducationField"]:
            if self.df is not None and candidate in self.df.columns:
                cat_col = candidate
                break

        if cat_col and self.df is not None:
            hue_col = target_selection if (target_selection in self.df.columns and target_selection != "-- Auto Detect --") else "Attrition_Prediction"
            
            # Group data for multi-bar plot
            grouped = self.df.groupby([cat_col, hue_col]).size().unstack(fill_value=0)
            grouped.plot(kind="bar", ax=ax4, color=["#377eb8", "#ff7f00"], width=0.8)
            
            ax4.set_title(f"Attrition by {cat_col}", fontsize=11, fontweight="bold")
            ax4.set_xlabel(cat_col)
            ax4.set_ylabel("Count")
            ax4.tick_params(axis="x", rotation=35)
            ax4.legend(title="Attrition", loc="upper right")
        else:
            ax4.text(0.5, 0.5, "Categorical Attributes\nNot Found in Input CSV", ha="center", va="center")
            ax4.set_title("Attribute Distribution Analysis", fontsize=11, fontweight="bold")

        ax4.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)