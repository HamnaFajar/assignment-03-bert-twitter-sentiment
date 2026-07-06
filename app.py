"""
Assignment 03 - PyQt GUI for BERT Twitter Sentiment Analysis
--------------------------------------------------------------
Run: python app.py

Workflow:
  1. Click "Load Dataset" -> pick your Twitter CSV -> tweets show in table
  2. Click "Load Model"   -> pick the saved_bert_model/ folder
  3. Click any row in the table -> prediction shows instantly
  4. Or type your own sentence in the box and click "Predict"
"""

import sys
import os
import json

import pandas as pd
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification

from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout,
    QTableWidget, QTableWidgetItem, QFileDialog, QLineEdit, QMessageBox,
    QHeaderView
)
from PyQt5.QtCore import Qt


class SentimentApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BERT Twitter Sentiment Analyzer")
        self.resize(800, 600)

        self.model = None
        self.tokenizer = None
        self.id2label = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.df = None
        self.text_col = None

        self.init_ui()

    def init_ui(self):
        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e2f;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QLabel#title {
                color: #ffffff;
                font-size: 22px;
                font-weight: bold;
                padding: 12px;
            }
            QLabel#status {
                color: #a0a0c0;
                font-size: 13px;
                padding: 6px;
            }
            QPushButton {
                background-color: #6c5ce7;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 16px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #8172f2;
            }
            QPushButton:pressed {
                background-color: #5643d1;
            }
            QPushButton#predictBtn {
                background-color: #00b894;
            }
            QPushButton#predictBtn:hover {
                background-color: #00d1a7;
            }
            QLineEdit {
                background-color: #2a2a40;
                color: white;
                border: 2px solid #6c5ce7;
                border-radius: 8px;
                padding: 8px;
                font-size: 14px;
            }
            QTableWidget {
                background-color: #2a2a40;
                color: #e0e0f0;
                border: none;
                border-radius: 8px;
                gridline-color: #3d3d5c;
                font-size: 13px;
            }
            QHeaderView::section {
                background-color: #6c5ce7;
                color: white;
                font-weight: bold;
                padding: 6px;
                border: none;
            }
            QTableWidget::item:selected {
                background-color: #6c5ce7;
            }
        """)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(12)

        # --- Title ---
        title = QLabel("BERT Twitter Sentiment Analyzer")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)

        # --- Top buttons ---
        top_row = QHBoxLayout()
        self.load_dataset_btn = QPushButton("Load Dataset")
        self.load_dataset_btn.clicked.connect(self.load_dataset)
        self.load_model_btn = QPushButton("Load Model")
        self.load_model_btn.clicked.connect(self.load_model)
        top_row.addWidget(self.load_dataset_btn)
        top_row.addWidget(self.load_model_btn)
        main_layout.addLayout(top_row)

        # --- Status label ---
        self.status_label = QLabel("Status: No dataset or model loaded yet.")
        self.status_label.setObjectName("status")
        main_layout.addWidget(self.status_label)

        # --- Table of tweets ---
        self.table = QTableWidget()
        self.table.setColumnCount(1)
        self.table.setHorizontalHeaderLabels(["Tweet / Sentence"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.cellClicked.connect(self.on_row_clicked)
        main_layout.addWidget(self.table)

        # --- Manual input ---
        manual_row = QHBoxLayout()
        self.manual_input = QLineEdit()
        self.manual_input.setPlaceholderText("Type a sentence to predict sentiment...")
        self.predict_btn = QPushButton("Predict")
        self.predict_btn.setObjectName("predictBtn")
        self.predict_btn.clicked.connect(self.predict_manual)
        manual_row.addWidget(self.manual_input)
        manual_row.addWidget(self.predict_btn)
        main_layout.addLayout(manual_row)

        # --- Prediction output ---
        self.result_label = QLabel("Prediction: -")
        self.result_label.setStyleSheet("""
            font-size: 20px;
            font-weight: bold;
            padding: 14px;
            color: white;
            background-color: #2a2a40;
            border-radius: 10px;
        """)
        self.result_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.result_label)

        self.setLayout(main_layout)

    # ------------------------------------------------------------
    def load_dataset(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Twitter CSV Dataset", "", "CSV Files (*.csv)")
        if not path:
            return
        try:
            self.df = pd.read_csv(path, encoding="latin-1")
            # try to auto-detect the text column
            candidates = [c for c in self.df.columns if c.lower() in
                          ("text", "tweet", "tweets", "sentence")]
            self.text_col = candidates[0] if candidates else self.df.columns[0]

            self.table.setRowCount(0)
            for i, row in self.df.head(200).iterrows():
                r = self.table.rowCount()
                self.table.insertRow(r)
                self.table.setItem(r, 0, QTableWidgetItem(str(row[self.text_col])))

            self.status_label.setText(
                f"Status: Dataset loaded ({len(self.df)} rows, showing first 200). "
                f"Using column '{self.text_col}' as tweet text."
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not load dataset:\n{e}")

    # ------------------------------------------------------------
    def load_model(self):
        folder = QFileDialog.getExistingDirectory(self, "Select saved_bert_model folder")
        if not folder:
            return
        try:
            self.tokenizer = BertTokenizerFast.from_pretrained(folder)
            self.model = BertForSequenceClassification.from_pretrained(folder)
            self.model.to(self.device)
            self.model.eval()

            label_map_path = os.path.join(folder, "label_map.json")
            if os.path.exists(label_map_path):
                with open(label_map_path) as f:
                    raw_map = json.load(f)
                self.id2label = {int(k): v for k, v in raw_map.items()}
            else:
                # fallback generic labels
                num_labels = self.model.config.num_labels
                self.id2label = {i: f"class_{i}" for i in range(num_labels)}

            self.status_label.setText("Status: Model loaded successfully. Ready to predict.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not load model:\n{e}")

    # ------------------------------------------------------------
    def predict_sentiment(self, text):
        if self.model is None or self.tokenizer is None:
            QMessageBox.warning(self, "No model", "Please load the trained model first.")
            return None
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True, max_length=64
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            pred_id = int(torch.argmax(probs, dim=1).item())
            confidence = float(probs[0][pred_id])
        label = self.id2label.get(pred_id, str(pred_id))
        return label, confidence

    # ------------------------------------------------------------
    def show_result(self, label, conf):
        colors = {
            "positive": "#00b894",
            "negative": "#d63031",
            "neutral": "#fdcb6e",
        }
        bg = colors.get(str(label).lower(), "#6c5ce7")
        text_color = "#1e1e2f" if str(label).lower() == "neutral" else "white"
        self.result_label.setStyleSheet(f"""
            font-size: 20px;
            font-weight: bold;
            padding: 14px;
            color: {text_color};
            background-color: {bg};
            border-radius: 10px;
        """)
        self.result_label.setText(f"Prediction: {label.upper()}  ({conf*100:.1f}% confidence)")

    # ------------------------------------------------------------
    def on_row_clicked(self, row, column):
        text = self.table.item(row, 0).text()
        result = self.predict_sentiment(text)
        if result:
            label, conf = result
            self.show_result(label, conf)

    # ------------------------------------------------------------
    def predict_manual(self):
        text = self.manual_input.text().strip()
        if not text:
            QMessageBox.warning(self, "Empty input", "Please type a sentence first.")
            return
        result = self.predict_sentiment(text)
        if result:
            label, conf = result
            self.show_result(label, conf)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SentimentApp()
    window.show()
    sys.exit(app.exec_())