# Ultrasonic Audio Communication System

A Python-based system for transmitting and receiving binary data (images or text) over ultrasonic audio (20–24 kHz) using multi-channel FSK, convolutional error correction, and simple framing.  
Ideal for short-range, line-of-sight data links using standard speakers and microphones.

---

## 🚀 Features

- **Multi‐Channel FSK**  
  Parallel frequency‐shift keying channels for higher throughput.

- **Convolutional ECC**  
  Rate-½ convolutional encoding (`G = [111, 101]`) with Viterbi decoding for bit‐error resilience.

- **Preamble Framing**  
  Robust pad-wave detection (fade‐in/out) for packet synchronization via cross‐correlation.

- **Image & Text Support**  
  Send black-and-white images or text as bitstreams.

- **Spectrogram Debugging**  
  Built-in spectrogram plotting for full recordings and extracted message segments.

- **Modular Design**  
  Separate modules for audio I/O, signal processing, error correction and compression.

---

## 📚 Table of Contents

- [Installation](#installation)  
- [Usage](#usage)  
  - [Transmitter](#transmitter)  
  - [Receiver](#receiver)  
- [Project Structure](#project-structure)  
- [Configuration](#configuration)  
- [Dependencies](#dependencies)  

---

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/Prom-DSP.git
   cd Prom-DSP
2. **Create a virtual environment & install dependencies**
   Or manually install:
   ```bash
   pip install numpy scipy matplotlib sounddevice pillow

---

## ▶️ Usage

### Transmitter

1. In `transmitter.py`, choose whether to send an image or text:
   ```python
   def main():
       # — Choose one —
       send_image("./images/eye3.jpg")
       # send_text("Hello, this is a test message.")
2. Run the transmitter script:
   python transmitter.py
The console will display bit counts and timestamps.
The system will emit the ultrasonic signal via your speakers.

### Receiver

1. In receiver.py, uncomment the appropriate handler for the expected message type:
   ```python
   def main():
      msg_sig = receive_audio_segment()
      # — Choose one —
      receive_image(msg_sig)
      # receive_text(msg_sig)
2. Run the receiver script:
```bash
   python receiver.py
The script records audio via your default microphone.
It detects pad markers, decodes the bitstream, and:
- Saves/displays an image, or
- Prints the decoded text message to the console.

---

## 🗂️ Project Structure

├── audio_processing.py   # Audio I/O, FSK modulation/demodulation, spectrograms
├── data_management.py    # Error-correcting encoder/decoder, framing logic
├── utils.py              # Bit/Image/Text conversion, compression utilities
├── consts.py             # Constants: sample rate, frequencies, pads, etc.
├── transmitter.py        # CLI entry point for sending image/text
├── receiver.py           # CLI entry point for receiving image/text
└── requirements.txt      # Dependency list

---

## ⚙️ Configuration

All signal parameters are configurable via `consts.py`:

- `SAMPLE_RATE` – Audio sampling rate (Hz)
- `BIT_DURATION` – Duration of a bit (seconds)
- `N_CHANNELS` – Number of simultaneous FSK tones per frame
- `FREQ_MIN`, `FREQ_MAX` – Frequency band for encoding (Hz)
- `PAD` – List of frequency sequences used for pad-wave synchronization

Adjust these values to optimize for your specific hardware setup or environmental conditions.

---

## 📦 Dependencies

- Python 3.8 or higher
- [NumPy](https://numpy.org/)
- [SciPy](https://scipy.org/)
- [Matplotlib](https://matplotlib.org/)
- [sounddevice](https://python-sounddevice.readthedocs.io/) *(requires PortAudio)*
- [Pillow (PIL)](https://python-pillow.org/)

Install with:
```bash
pip install -r requirements.txt
