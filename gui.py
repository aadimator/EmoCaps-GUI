import sys
import numba

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtCore import QFileInfo, QObject, QThread, QUrl, pyqtSignal
from plotly.subplots import make_subplots
from PyQt5 import QtWidgets, QtWebEngineWidgets, uic

from preprocess import preprocess
from predict import predict, dominant_emotion, class_to_vad, vad_to_emotion

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class Worker(QObject):
    preprocessd = pyqtSignal()
    finished = pyqtSignal(np.ndarray)

    def __init__(self, filename, start_time, end_time, parent=None) -> None:
        super().__init__(parent=parent)
        self.filename = filename
        self.start_time = start_time
        self.end_time = end_time

    def run(self):
        """Long-running task."""
        preprocess(self.filename, "preprocessed.dat", experiment_start=self.start_time, experiment_end=self.end_time)
        self.preprocessd.emit()
        pred = predict("preprocessed.dat", "bin\\sub_ind\\best_fold.h5")
        self.finished.emit(pred)

class EmoCapsApp(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        QtWidgets.QMainWindow.__init__(self)
        self.ui = uic.loadUi('main.ui', self)
        self.resize(1280, 1080)
        self.browser = self.webEngineView

        self.filename = ''
        self.start_time = 3
        self.end_time = None
        self.max_emotion_freq = None

        url = QUrl(QFileInfo("media\emocaps.html").absoluteFilePath())
        self.browser.load(url)

        self.pushButtonFile.clicked.connect(self.getFile)
        self.lineEditStartTime.textChanged['QString'].connect(self.update_start_time)
        self.lineEditEndTime.textChanged['QString'].connect(self.update_end_time)
        self.lineEditFile.textChanged['QString'].connect(self.file_selected)
        self.pushButtonClassify.clicked.connect(self.classify)


    def getFile(self):
        """ This function will get the address of the csv file location
            also calls a readData function 
        """
        self.filename = QFileDialog.getOpenFileName(filter = "EDF (*.edf)")[0]
        print("File :", self.filename)
        self.lineEditFile.setText(self.filename)
        self.update_status("File read successfully.")

    def file_selected(self, value):
        self.pushButtonClassify.setEnabled(True)

    def update_start_time(self, value):
        self.start_time = int(value)

    def update_end_time(self, value):
        self.end_time = int(value)

    def update_status(self, text):
        self.labelStatus.setText(text)

    def classify(self):
        self.update_status("Pre-processing file ...")
        try:
            numba.njit(lambda x: x + 1)(123)
        except:
            pass

        url = QUrl(QFileInfo("media\loader.html").absoluteFilePath())
        self.browser.load(url)
        
        # Step 2: Create a QThread object
        self.thread = QThread()
        # Step 3: Create a worker object
        self.worker = Worker(self.filename, self.start_time, self.end_time)
        # Step 4: Move worker to the thread
        self.worker.moveToThread(self.thread)
        # Step 5: Connect signals and slots
        self.thread.started.connect(self.worker.run)
        self.worker.preprocessd.connect(self.update_status_preprocessed)
        self.worker.finished.connect(self.update_plots)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        # Step 6: Start the thread
        self.thread.start()


    def update_status_preprocessed(self):
        self.update_status("File preprocessing complete.")
        self.update_status("Starting Emotion Prediction")

    def update_plots(self, pred):
        print(pred)
        self.max_emotion_freq = np.bincount(pred).argmax()  # max freq
        self.update_status("Emotion Prediction completed.")
        emotion_labels = vad_to_emotion(class_to_vad(pred))
        self.labelResult.setText(f'<html><head/><body><p><span style=" font-size:24pt;">{dominant_emotion(emotion_labels)}</span></p></body></html>')

        df = pd.DataFrame(emotion_labels, columns=['Emotions'])
        fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.6, 0.4],
        subplot_titles=("Emotion Labels", "Emotion Frequency"),
        specs=[[{"type": "scatter"}, {"type": "bar"}]])

        fig.add_trace(
            go.Scatter(x=df.index, y=df.Emotions,
                            mode='lines+markers',
                            showlegend=False), row=1, col=1)

        fig.add_trace(
            go.Histogram(x=df.Emotions, showlegend=False),
            row=1, col=2
        )

        fig.update_layout(
            # template="plotly_dark",
            margin=dict(r=10, t=40, b=10, l=10),
            xaxis_rangeslider_visible=True
            # title_text='Emotion Labels' # title of plot
        )

        self.browser.setHtml(fig.to_html(include_plotlyjs='cdn'))

app = QtWidgets.QApplication(sys.argv)
mainWindow = EmoCapsApp()
mainWindow.show()
sys.exit(app.exec_())