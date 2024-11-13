import numpy as np
from scipy.signal import welch
from pylsl import StreamInlet, resolve_stream
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
import pyqtgraph as pg

class EEGAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EEG Analysis")
        
        # Create layout
        layout = QVBoxLayout()
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Create plots for different frequency bands
        self.raw_plot = pg.PlotWidget(title="Raw EEG")
        self.bands_plot = pg.PlotWidget(title="Frequency Bands Power")
        
        layout.addWidget(self.raw_plot)
        layout.addWidget(self.bands_plot)

        # Set up LSL inlet
        streams = resolve_stream('name', 'BioAmpDataStream')
        self.inlet = StreamInlet(streams[0])
        
        # Initialize data buffers
        self.buffer_size = 1000
        self.eeg_buffer = np.zeros((6, self.buffer_size))
        
        # Set up frequency bands
        self.bands = {
            'Delta': (0.5, 4),
            'Theta': (4, 8),
            'Alpha': (8, 13),
            'Beta': (13, 30),
            'Gamma': (30, 45)
        }
        
        # Set up timer for updates
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(100)  # Update every 100ms

    def calculate_band_powers(self, data, fs=250):
        """Calculate power in different frequency bands"""
        powers = {}
        freqs, psd = welch(data, fs=fs, nperseg=fs*2)
        
        for band, (low, high) in self.bands.items():
            # Find frequencies that correspond to the band
            idx = np.logical_and(freqs >= low, freqs <= high)
            # Calculate average power in the band
            powers[band] = np.mean(psd[idx])
            
        return powers

    def update(self):
        # Get new data
        samples, _ = self.inlet.pull_chunk()
        if samples:
            samples = np.array(samples)
            
            # Update buffer
            self.eeg_buffer = np.roll(self.eeg_buffer, -len(samples), axis=1)
            self.eeg_buffer[:, -len(samples):] = samples.T
            
            # Update raw plot
            self.raw_plot.clear()
            for i in range(6):
                self.raw_plot.plot(self.eeg_buffer[i], pen=(i,6))
            
            # Calculate and plot band powers
            self.bands_plot.clear()
            powers = self.calculate_band_powers(self.eeg_buffer[0])  # Analysis on first channel
            x = np.arange(len(powers))
            self.bands_plot.plot(x, list(powers.values()), pen='b')
            
            # Add band labels
            ax = self.bands_plot.getAxis('bottom')
            ax.setTicks([[(i, band) for i, band in enumerate(powers.keys())]])

def main():
    app = QApplication([])
    window = EEGAnalyzer()
    window.show()
    app.exec_()

if __name__ == '__main__':
    main() 