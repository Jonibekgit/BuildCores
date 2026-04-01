

import pyaudio
import numpy as np
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import sys
import collections


RATE = 44100          
CHUNK = 1024          
FORMAT = pyaudio.paFloat32
CHANNELS = 1

try:
    pa = pyaudio.PyAudio()

    device_index = None
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            device_index = i
            print(f"Using mic: {info['name']}")
            break

    if device_index is None:
        print("ERROR: No microphone found.")
        print("Check your system audio settings.")
        sys.exit(1)

    stream = pa.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=CHUNK,
    )
    print("Mic stream opened successfully.")

except Exception as e:
    print(f"ERROR opening microphone: {e}")
    print("\nFixes:")
    print("  Mac:   brew install portaudio && pip install pyaudio")
    print("  Linux: sudo apt-get install portaudio19-dev && pip install pyaudio")
    print("  Win:   pip install pipwin && pipwin install pyaudio")
    sys.exit(1)

CUTOFF_HZ = 0.5       
FILTER_ORDER = 2

effective_rate = RATE / CHUNK  ]
nyquist = effective_rate / 2

normalized_cutoff = min(CUTOFF_HZ / nyquist, 0.95)
b_coeff, a_coeff = butter(FILTER_ORDER, normalized_cutoff, btype='low')


BREATH_THRESHOLD = 0.005  


HISTORY_LENGTH = 500    

raw_history = collections.deque([0.0] * HISTORY_LENGTH, maxlen=HISTORY_LENGTH)

envelope_history = collections.deque([0.0] * HISTORY_LENGTH, maxlen=HISTORY_LENGTH)

filter_state = np.zeros(max(len(a_coeff), len(b_coeff)) - 1)

breath_times = []           
is_above_threshold = False  
current_bpm = 0.0

breath_count = 0

BPM_SMOOTHING_WINDOW = 5
bpm_history = collections.deque(maxlen=BPM_SMOOTHING_WINDOW)


def compute_bpm():
    """Calculate breaths per minute from recent breath timestamps,
    then smooth the result using a rolling average."""
    now = time.time()

    recent = [t for t in breath_times if now - t < 30]
    breath_times.clear()
    breath_times.extend(recent)

    if len(recent) < 2:
        return 0.0

    intervals = [recent[i+1] - recent[i] for i in range(len(recent)-1)]
    avg_interval = sum(intervals) / len(intervals)

    if avg_interval > 0:
        raw_bpm = 60.0 / avg_interval

        bpm_history.append(raw_bpm)
        return sum(bpm_history) / len(bpm_history)

    return 0.0

plt.style.use('dark_background')   # sets default dark colors for all elements

BG_COLOR      = '#0d0d0d'   # near-black for figure and axes backgrounds
PANEL_COLOR   = '#1a1a2e'   # slightly lighter panel color for axes
CYAN          = '#00e5ff'   # raw waveform line
GREEN         = '#69ff47'   # envelope line and BPM text
RED           = '#ff1744'   # threshold dashed line
WHITE         = '#ffffff'   # status text and axis labels

fig, (ax_raw, ax_env) = plt.subplots(2, 1, figsize=(10, 6))

fig.patch.set_facecolor(BG_COLOR)
for ax in (ax_raw, ax_env):
    ax.set_facecolor(PANEL_COLOR)
    ax.tick_params(colors=WHITE)
    ax.xaxis.label.set_color(WHITE)
    ax.yaxis.label.set_color(WHITE)
    ax.title.set_color(WHITE)
    for spine in ax.spines.values():
        spine.set_edgecolor('#333355')   # subtle border color

fig.suptitle("BreathClock — Day 06", fontsize=14, fontweight='bold', color=WHITE)

ax_raw.set_xlim(0, HISTORY_LENGTH)
ax_raw.set_ylim(0, 0.05)
ax_raw.set_ylabel("Raw Amplitude", color=WHITE)
ax_raw.set_title("Mic Input (RMS per chunk)", color=WHITE)
line_raw, = ax_raw.plot([], [], color=CYAN, linewidth=1)

ax_env.set_xlim(0, HISTORY_LENGTH)
ax_env.set_ylim(0, 0.03)
ax_env.set_ylabel("Envelope", color=WHITE)
ax_env.set_xlabel("Time →", color=WHITE)
ax_env.set_title("Filtered Breath Envelope", color=WHITE)
line_env, = ax_env.plot([], [], color=GREEN, linewidth=2)

threshold_line = ax_env.axhline(y=BREATH_THRESHOLD, color=RED,
                                 linestyle='--', linewidth=1, label='Threshold')
ax_env.legend(loc='upper right', facecolor=PANEL_COLOR, edgecolor='#333355',
              labelcolor=WHITE)

bpm_text = ax_env.text(0.02, 0.85, "BPM: --", transform=ax_env.transAxes,
                        fontsize=16, fontweight='bold', color=GREEN,
                        verticalalignment='top')

breath_count_text = ax_raw.text(
    0.02, 0.60,             # x=2% from left, y=60% up the axes
    "Breaths: 0",
    transform=ax_raw.transAxes,
    fontsize=12,
    fontweight='bold',
    color=CYAN,
    verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor=BG_COLOR, alpha=0.7)
)

status_text = ax_raw.text(0.02, 0.85, "Waiting...", transform=ax_raw.transAxes,
                           fontsize=12, color=WHITE,
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor=BG_COLOR, alpha=0.7))

plt.tight_layout()


def update(frame_num):
    global filter_state, is_above_threshold, current_bpm, breath_count

    try:

        audio_data = stream.read(CHUNK, exception_on_overflow=False)
        samples = np.frombuffer(audio_data, dtype=np.float32)

        rms = np.sqrt(np.mean(samples ** 2))
        raw_history.append(rms)

        raw_array = np.array(raw_history)
        filtered, filter_state = lfilter(b_coeff, a_coeff, [rms], zi=filter_state)
        envelope_val = abs(filtered[0])
        envelope_history.append(envelope_val)


        if envelope_val > BREATH_THRESHOLD and not is_above_threshold:
            is_above_threshold = True
            breath_times.append(time.time())
            current_bpm = compute_bpm()

            breath_count += 1

        elif envelope_val < BREATH_THRESHOLD * 0.7:  # Hysteresis!
            is_above_threshold = False

        x_data = list(range(HISTORY_LENGTH))
        line_raw.set_data(x_data, list(raw_history))
        line_env.set_data(x_data, list(envelope_history))

        raw_max = max(list(raw_history)[-100:]) if any(raw_history) else 0.01
        env_max = max(list(envelope_history)[-100:]) if any(envelope_history) else 0.01
        ax_raw.set_ylim(0, max(raw_max * 1.5, 0.005))
        ax_env.set_ylim(0, max(env_max * 1.5, BREATH_THRESHOLD * 2))

        threshold_line.set_ydata([BREATH_THRESHOLD])

        if current_bpm > 0:
            bpm_text.set_text(f"BPM: {current_bpm:.1f}")
        else:
            bpm_text.set_text("BPM: -- (breathe near mic)")
        breath_count_text.set_text(f"Breaths: {breath_count}")

        breathing = "BREATH DETECTED" if is_above_threshold else "Listening..."
        breath_color = GREEN if is_above_threshold else WHITE
        status_text.set_text(breathing)
        status_text.set_color(breath_color)

    except Exception as e:
        status_text.set_text(f"Error: {e}")

    return line_raw, line_env, bpm_text, status_text, threshold_line, breath_count_text


print("\nBreathClock is running!")
print(f"Filter cutoff: {CUTOFF_HZ} Hz")
print(f"Breath threshold: {BREATH_THRESHOLD}")
print(f"Effective sample rate: {effective_rate:.1f} Hz")
print(f"BPM smoothing window: {BPM_SMOOTHING_WINDOW} readings")
print("\nBreathe slowly and deliberately near your mic.")
print("Watch the green envelope — each peak is a breath.")
print("Close the plot window to quit.\n")

try:
    ani = animation.FuncAnimation(fig, update, interval=int(1000 * CHUNK / RATE),
                                   blit=False, cache_frame_data=False)
    plt.show()
except KeyboardInterrupt:
    pass
finally:
    stream.stop_stream()
    stream.close()
    pa.terminate()
    print(f"\nBreathClock ended.")
    print(f"Final BPM   : {current_bpm:.1f}")
    print(f"Total breaths: {breath_count}")
    print("See you tomorrow for Day 07!")
