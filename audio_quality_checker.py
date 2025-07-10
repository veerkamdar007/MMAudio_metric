import numpy as np
import librosa
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

def load_audio(file_path, sr=22050):
    """Load audio file"""
    try:
        audio, _ = librosa.load(file_path, sr=sr)
        return audio
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def clarity_score(audio, sr):
    
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
    spectral_flatness = librosa.feature.spectral_flatness(y=audio)[0]
    
    centroid_mean = np.mean(spectral_centroid)
    flatness_mean = np.mean(spectral_flatness)
    
    # Centroid should be in a good range (not too low/high)
    centroid_score = 1.0 - abs(centroid_mean - sr/4) / (sr/2)  # Penalize extreme values
    centroid_score = max(0, centroid_score)
    
    # Flatness should indicate some spectral richness but not be too flat
    flatness_score = min(1.0, flatness_mean * 3)  # Some spectral variation is good
    
    return (centroid_score + flatness_score) / 2

def loudness_consistency_score(audio):

    # Get RMS energy over time
    rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
    rms_db = 20 * np.log10(rms + 1e-10)
    
    # Good audio has reasonable loudness variation
    loudness_std = np.std(rms_db)
    
    # Score based on reasonable variation 
    if loudness_std < 5: 
        consistency_score = loudness_std / 5
    elif loudness_std > 25:  
        consistency_score = max(0, 1 - (loudness_std - 25) / 25)
    else:  
        consistency_score = 1.0
    
    return consistency_score

def distortion_detection_score(audio):
    """Detect various distortions that affect perceived quality"""
    # Check for clipping
    clipping_ratio = np.sum(np.abs(audio) > 0.95) / len(audio)
    clipping_score = 1.0 - min(1.0, clipping_ratio * 50)  # Heavy penalty for clipping
    
    # Check peak levels
    peak_level = np.max(np.abs(audio))
    if peak_level > 1.0:  
        peak_score = 0.0
    elif peak_level < 0.1:  
        peak_score = peak_level * 10
    else:
        peak_score = 1.0
    
    
    dc_offset = abs(np.mean(audio))
    dc_score = 1.0 - min(1.0, dc_offset * 100)
    
    return (clipping_score + peak_score + dc_score) / 3

def dynamic_range_score(audio):
    
    rms = librosa.feature.rms(y=audio)[0]
    rms_db = 20 * np.log10(rms + 1e-10)
    
    # Calculate dynamic range
    dynamic_range = np.percentile(rms_db, 95) - np.percentile(rms_db, 5)
    
    # Good audio has reasonable dynamic range
    if dynamic_range < 10:  # Too compressed
        dr_score = dynamic_range / 10
    elif dynamic_range > 40:  # Too much variation
        dr_score = max(0, 1 - (dynamic_range - 40) / 20)
    else:  # Good range
        dr_score = 1.0
    
    return dr_score

def naturalness_score(audio, sr):
    """Assess overall naturalness using spectral and temporal features"""
    # Spectral features
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
    zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)[0]
    
    # Temporal features
    onset_frames = librosa.onset.onset_detect(y=audio, sr=sr)
    onset_density = len(onset_frames) / (len(audio) / sr)
    
    # Natural audio has reasonable spectral bandwidth
    bandwidth_mean = np.mean(spectral_bandwidth)
    bandwidth_score = min(1.0, bandwidth_mean / (sr/4)) 
    
    # Natural audio has reasonable zero crossing rate
    zcr_mean = np.mean(zero_crossing_rate)
    if zcr_mean < 0.01:  
        zcr_score = zcr_mean * 100
    elif zcr_mean > 0.3:  
        zcr_score = max(0, 1 - (zcr_mean - 0.3) * 2)
    else:
        zcr_score = 1.0
    
    # Onset density should be reasonable
    if onset_density > 10:  # Too many onsets
        onset_score = max(0, 1 - (onset_density - 10) / 10)
    else:
        onset_score = min(1.0, onset_density / 5)  # Some activity is good
    
    return (bandwidth_score + zcr_score + onset_score) / 3

def create_visualizations(audio, sr, results, filename):

    plt.style.use('default')
    sns.set_palette("husl")
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f'Audio Quality Analysis: {Path(filename).name}', fontsize=16, fontweight='bold', y=0.98)
    
    ax1 = plt.subplot(2, 3, 1)
    score = results['overall_score']

    theta = np.linspace(0, np.pi, 100)
    r = 1
    ax1.plot(r * np.cos(theta), r * np.sin(theta), 'lightgray', linewidth=8)
    
    score_theta = np.linspace(0, np.pi * (score/100), int(max(1, score)))
    if score >= 80:
        color = 'green'
    elif score >= 60:
        color = 'orange'
    else:
        color = 'red'
    
    ax1.plot(r * np.cos(score_theta), r * np.sin(score_theta), color, linewidth=8)
    ax1.text(0, -0.3, f'{score:.1f}/100', ha='center', va='center', fontsize=18, fontweight='bold')
    ax1.text(0, -0.5, 'Overall Quality', ha='center', va='center', fontsize=10)
    
    ax1.set_xlim(-1.2, 1.2)
    ax1.set_ylim(-0.7, 1.2)
    ax1.set_aspect('equal')
    ax1.axis('off')
    
    # 2. Component Scores Bar Chart
    ax2 = plt.subplot(2, 3, 2)
    components = ['Clarity', 'Loudness', 'Distortion', 'Dynamic', 'Natural']
    scores = [results['clarity'], results['loudness_consistency'], results['distortion_free'], 
              results['dynamic_range'], results['naturalness']]
    
    colors = ['green' if s >= 70 else 'orange' if s >= 50 else 'red' for s in scores]
    bars = ax2.bar(components, scores, color=colors, alpha=0.7, width=0.6)
    
    # Add score labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{score:.0f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    ax2.set_ylim(0, 110)
    ax2.set_ylabel('Score', fontsize=10)
    ax2.set_title('Quality Components', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', labelsize=9)
    ax2.tick_params(axis='y', labelsize=9)
    
    # 3. Waveform
    ax3 = plt.subplot(2, 3, 3)
    time = np.linspace(0, len(audio)/sr, len(audio))
    ax3.plot(time, audio, alpha=0.7, linewidth=0.5)
    ax3.set_xlabel('Time (s)', fontsize=10)
    ax3.set_ylabel('Amplitude', fontsize=10)
    ax3.set_title('Waveform', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(labelsize=9)
    
    # 4. Spectrogram
    ax4 = plt.subplot(2, 3, 4)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    img = librosa.display.specshow(D, y_axis='hz', x_axis='time', sr=sr, ax=ax4)
    ax4.set_title('Spectrogram', fontsize=12)
    ax4.tick_params(labelsize=9)
    cbar = plt.colorbar(img, ax=ax4, format='%+2.0f dB')
    cbar.ax.tick_params(labelsize=8)
    
    # 5. RMS Energy over time
    ax5 = plt.subplot(2, 3, 5)
    rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=512)
    ax5.plot(times, rms, linewidth=1.5)
    ax5.set_xlabel('Time (s)', fontsize=10)
    ax5.set_ylabel('RMS Energy', fontsize=10)
    ax5.set_title('Loudness over Time', fontsize=12)
    ax5.grid(True, alpha=0.3)
    ax5.tick_params(labelsize=9)
    
    # 6. Frequency Spectrum
    ax6 = plt.subplot(2, 3, 6)
    fft = np.fft.fft(audio)
    magnitude = np.abs(fft)
    freqs = np.fft.fftfreq(len(fft), 1/sr)
    
    # Only plot positive frequencies
    pos_mask = freqs > 0
    ax6.semilogx(freqs[pos_mask], 20 * np.log10(magnitude[pos_mask] + 1e-10), linewidth=1)
    ax6.set_xlabel('Frequency (Hz)', fontsize=10)
    ax6.set_ylabel('Magnitude (dB)', fontsize=10)
    ax6.set_title('Frequency Spectrum', fontsize=12)
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim(20, sr/2)
    ax6.tick_params(labelsize=9)
    
    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the plot
    output_file = 'audio_quality_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    
    # Data for radar chart
    categories = ['Clarity', 'Loudness\nConsistency', 'Distortion\nFree', 'Dynamic\nRange', 'Naturalness']
    values = [results['clarity'], results['loudness_consistency'], results['distortion_free'], 
              results['dynamic_range'], results['naturalness']]
    
    # Number of variables
    N = len(categories)
    
    # Compute angle for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    # Add the first value at the end to close the circle
    values += values[:1]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    # Plot the values
    ax.plot(angles, values, 'o-', linewidth=2, label='Quality Scores')
    ax.fill(angles, values, alpha=0.25)
    
    # Add category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'])
    ax.grid(True)
    
    plt.title('Audio Quality Radar Chart', size=16, fontweight='bold', pad=20)
    
    # Add quality zones
    ax.fill_between(angles, 0, 50, alpha=0.1, color='red', label='Poor')
    ax.fill_between(angles, 50, 70, alpha=0.1, color='orange', label='Fair')
    ax.fill_between(angles, 70, 100, alpha=0.1, color='green', label='Good')
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    # Save the radar chart
    radar_file = 'audio_quality_radar.png'
    plt.savefig(radar_file, dpi=300, bbox_inches='tight')
    print(f"Radar chart saved as: {radar_file}")
    
    plt.show()

def evaluate_audio_quality(audio_path):
    print(f"Loading audio: {audio_path}")
    audio = load_audio(audio_path)
    
    if audio is None:
        return None, None
    
    sr = 22050  # Sample rate used for loading
    
    print("Analyzing audio quality...")
    
    clarity = clarity_score(audio, sr)
    loudness = loudness_consistency_score(audio)
    distortion = distortion_detection_score(audio)
    dynamic_range = dynamic_range_score(audio)
    naturalness = naturalness_score(audio, sr)
    
    # Weighted overall score (emphasize most important factors,these weights could be changed too)
    weights = {
        'clarity': 0.25,
        'loudness': 0.2,
        'distortion': 0.3,  # Most important - distortion ruins quality
        'dynamic_range': 0.15,
        'naturalness': 0.1
    }
    
    overall_score = (
        weights['clarity'] * clarity +
        weights['loudness'] * loudness +
        weights['distortion'] * distortion +
        weights['dynamic_range'] * dynamic_range +
        weights['naturalness'] * naturalness
    ) * 100
    
    results = {
        'overall_score': overall_score,
        'clarity': clarity * 100,
        'loudness_consistency': loudness * 100,
        'distortion_free': distortion * 100,
        'dynamic_range': dynamic_range * 100,
        'naturalness': naturalness * 100
    }
    
    return results, (audio, sr)

def print_results(results, filename):
    if results is None:
        print("Error: Could not evaluate audio quality")
        return
    
    print("\n" + "="*50)
    print("AUDIO QUALITY ASSESSMENT")
    print("="*50)
    print(f"File: {filename}")
    print()
    
    score = results['overall_score']
    print(f"Overall Quality Score: {score:.1f}/100")
    
    
    if score >= 85:
        category = "Excellent - Professional quality"
    elif score >= 75:
        category = "Very Good - High quality"
    elif score >= 65:
        category = "Good - Acceptable quality"
    elif score >= 50:
        category = "Fair - Some issues present"
    else:
        category = "Poor - Significant quality issues"
    
    print(f"Quality Assessment: {category}")
    print()
    
    # Component scores
    print("Quality Components:")
    print("-" * 35)
    components = [
        ("Clarity", results['clarity']),
        ("Loudness Consistency", results['loudness_consistency']),
        ("Distortion-Free", results['distortion_free']),
        ("Dynamic Range", results['dynamic_range']),
        ("Naturalness", results['naturalness'])
    ]
    
    for name, score in components:
        print(f"{name:18}: {score:5.1f}/100")
    
    print("="*50)
    
    # Recommendations
    print("Assessment:")
    if score >= 80:
        print("High quality audio suitable for professional use")
    elif score >= 65:
        print("Good quality audio with minor imperfections")
    elif score >= 50:
        print("Acceptable quality but could be improved")
    else:
        print("Quality issues detected - improvement recommended")
        
        # Specific recommendations
        if results['distortion_free'] < 60:
            print("  - Reduce distortion and clipping")
        if results['clarity'] < 60:
            print("  - Improve frequency balance")
        if results['dynamic_range'] < 60:
            print("  - Adjust dynamic range")
    
    print("="*50)

def main():
    audio_file = "gen_audio/fruit_splash_test_result.wav" # The audio file we want the metrics for.

    if not Path(audio_file).exists():
        print(f"Error: Audio file not found: {audio_file}")
        sys.exit(1)
    
    # Evaluate quality
    results, audio_data = evaluate_audio_quality(audio_file)
    
    if results is None:
        print("Failed to analyze audio quality")
        sys.exit(1)
    
    print_results(results, audio_file)

    print("\nGenerating visualizations...")
    audio, sr = audio_data
    
    create_visualizations(audio, sr, results, audio_file)
    create_quality_radar_chart(results)
    
    print("\nAnalysis complete! Check the generated PNG files for visual results.")

if __name__ == "__main__":
    main()
