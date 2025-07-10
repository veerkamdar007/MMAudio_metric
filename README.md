# Audio Quality Assessment & Visualization Tool

This project is a **simple yet insightful tool** for evaluating the perceptual quality of audio files — especially **generated background music**.  
It calculates various perceptual metrics and generates clear visualizations to help you understand the quality and potential issues in your audio.

---

## What does it do?

 --> Loads a `.wav` audio file and analyzes:  
- **Clarity** — Is the frequency content balanced and clear?  
- **Loudness Consistency** — Is the volume steady or wildly varying?  
- **Distortion-Free** — Checks for clipping, peaks, and DC offset.  
- **Dynamic Range** — Measures if your audio is too compressed or too variable.  
- **Naturalness** — Checks spectral and temporal features for a human-like feel.

--> Computes an **Overall Quality Score** (0–100) with an easy-to-understand interpretation.

--> Generates **visual reports**, including:
- Overall score gauge.
- Component scores bar chart.
- Waveform, spectrogram, RMS plot.
- Frequency spectrum.
- A clear **radar chart** summarizing all metrics.
