import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, font
import pandas as pd

# === Global variables ===
last_result = {"indices": None, "samples": None, "fs": None}
signal_data = {"signal": None, "fs": None}
quant_results = {
    "interval_indices": None, "encoded_signal": None,
    "quantized_signal": None, "error_signal": None
}
save_quant_btn = None

# --- Custom Animation & Styling Class ---
class AnimatedButton(tk.Button):
    """A custom button with hover animations for a modern feel."""
    def __init__(self, master=None, **kwargs):
        self.normal_bg = kwargs.get("background", "#2c2c2c")
        self.hover_bg = kwargs.get("hoverbackground", "#007bff")
        self.active_bg = kwargs.get("activebackground", "#0056b3")
        
        # Remove custom keys from kwargs before passing to parent
        kwargs.pop("hoverbackground", None)
        kwargs.pop("activebackground", None)

        super().__init__(master, **kwargs)
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)
        self.bind("<Button-1>", self.on_press)
        self.bind("<ButtonRelease-1>", self.on_release)

    def on_enter(self, e):
        self.config(background=self.hover_bg)

    def on_leave(self, e):
        self.config(background=self.normal_bg)
    
    def on_press(self, e):
        self.config(background=self.active_bg)

    def on_release(self, e):
        if self.winfo_containing(e.x_root, e.y_root) == self:
            self.config(background=self.hover_bg)
        else:
            self.config(background=self.normal_bg)


# --- Quantization Logic ---
def perform_quantization(signal, fs, levels=None, bits=None):
    if levels is None and bits is None:
        messagebox.showerror("Error", "Either levels or bits must be provided.")
        return
    if bits is not None:
        levels = 2 ** bits
    else:
        bits = int(np.ceil(np.log2(levels)))
    Vmin, Vmax = np.min(signal), np.max(signal)
    if Vmax == Vmin:
        quantized_signal = np.full_like(signal, Vmin)
        error_signal = np.zeros_like(signal)
        encoded_signal = [format(0, f'0{bits}b')] * len(signal)
        interval_indices = np.ones(len(signal), dtype=int)
    else:
        delta = (Vmax - Vmin) / levels
        midpoints = Vmin + (np.arange(levels) + 0.5) * delta
        quantized_signal = np.zeros_like(signal)
        encoded_indices = np.zeros(len(signal), dtype=int)
        interval_indices = np.zeros(len(signal), dtype=int)
        for i, sample in enumerate(signal):
            index = np.clip(int(np.floor((sample - Vmin) / delta)), 0, levels - 1)#index level
            quantized_signal[i] = midpoints[index]
            encoded_indices[i] = index
            interval_indices[i] = index + 1
        error_signal = signal - quantized_signal
        encoded_signal = [format(idx, f'0{bits}b') for idx in encoded_indices]
    display_quantization_results(signal, quantized_signal, error_signal, encoded_signal, interval_indices, fs, bits)

# --- Display & Data Handling ---
def  display_quantization_results(original, quantized, error, encoded, interval_indices, fs, bits):
    plt.style.use('dark_background')
    time_axis = np.linspace(0, len(original) / fs, len(original), endpoint=False)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f'Quantization Results ({2**bits} Levels, {bits} Bits)', fontsize=16, color='white')
    fig.patch.set_facecolor('#212121')

    ax1.set_facecolor('#2c2c2c')
    ax1.plot(time_axis, original, 'c-', label='Original Signal', alpha=0.8)
    ax1.step(time_axis, quantized, '#ff8f00', where='mid', label='Quantized Signal')
    ax1.set_title('Original vs. Quantized Signal', color='white')
    ax1.set_ylabel('Amplitude', color='white')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.tick_params(axis='x', colors='white')
    ax1.tick_params(axis='y', colors='white')

    ax2.set_facecolor('#2c2c2c')
    ax2.plot(time_axis, error, 'g-', label='Quantization Error')
    ax2.set_title('Quantization Error', color='white')
    ax2.set_xlabel('Time (s)', color='white')
    ax2.set_ylabel('Error', color='white')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.tick_params(axis='x', colors='white')
    ax2.tick_params(axis='y', colors='white')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    show_encoded_data(original, quantized, encoded, interval_indices, error)
    global quant_results, save_quant_btn
    quant_results.update({
        "interval_indices": interval_indices, "encoded_signal": encoded,
        "quantized_signal": quantized, "error_signal": error
    })
    if save_quant_btn:
        save_quant_btn.config(state=tk.NORMAL)

def show_encoded_data(original, quantized, encoded, interval_indices, error):
    data_win = tk.Toplevel()
    data_win.title("Encoded Signal Data (First 100 Samples)")
    data_win.geometry("800x450")
    data_win.configure(bg="#212121")

    style = ttk.Style(data_win)
    style.theme_use("clam")
    style.configure("Treeview", background="#2c2c2c", foreground="#f0f0f0", fieldbackground="#2c2c2c", rowheight=25)
    style.configure("Treeview.Heading", background="#3c3c3c", foreground="#007bff", font=('Segoe UI', 10, 'bold'))
    style.map('Treeview.Heading', background=[('active', '#5c5c5c')])

    cols = ('Sample Index', 'Interval', 'Original', 'Quantized', 'Encoded', 'Error')
    tree = ttk.Treeview(data_win, columns=cols, show='headings')
    for col in cols:
        tree.heading(col, text=col)
        tree.column(col, width=130, anchor='center')
    for i, (orig, quant, enc, interval, err) in enumerate(zip(original, quantized, encoded, interval_indices, error)):
        if i >= 100: break
        tree.insert("", "end", values=(i, interval, f"{orig:.4f}", f"{quant:.4f}", enc, f"{err:.4f}"))
    tree.pack(expand=True, fill='both', padx=10, pady=10)

def generate_signal(wave_type, amplitude, phase_rad, analog_freq, sampling_freq, duration=1.0):
    if sampling_freq <= 0 or analog_freq <= 0:
        messagebox.showerror("Input Error", "Frequencies must be positive values.")
        return None, None
    if sampling_freq < 2 * analog_freq:
        messagebox.showwarning("Nyquist Warning", "Sampling frequency should be at least twice the analog frequency to avoid aliasing.")
    t = np.linspace(0, duration, int(sampling_freq * duration), endpoint=False)
    omega = 2 * np.pi * analog_freq
    signal = amplitude * (np.sin if wave_type == "Sine" else np.cos)(omega * t + phase_rad)
    return t, signal

# === File operations ===
def read_signal_file(file_path):
    try:
        data = pd.read_csv(file_path, sep='\s+', header=None, skiprows=3, names=['index', 'sample'], on_bad_lines='skip')
        indices = data['index'].to_numpy(dtype=int)
        samples = data['sample'].to_numpy(dtype=float)
        with open(file_path, "r") as f:
            lines = f.readlines()
            fs = int(lines[2].strip()) if lines[2].strip() else len(samples)
        return indices, samples, fs
    except Exception as e:
        messagebox.showerror("File Error", f"Could not read the file:\n{file_path}\n{e}")
        return None, None, None

def load_signal_from_file_dialog():
    file_path = filedialog.askopenfilename(title="Select a signal file", filetypes=[("Text files", "*.txt")])
    return read_signal_file(file_path) if file_path else (None, None, None)

def save_signal_to_txt(indices=None, signal=None, sampling_freq=None):
    if signal is None or indices is None:
        indices = last_result['indices']
        signal = last_result['samples']
        sampling_freq = last_result['fs']

    if signal is None:
        messagebox.showwarning("No Signal", "No result to save.")
        return
    
    if sampling_freq is None:
        sampling_freq = len(signal)
        
    file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
    if not file_path: return
    with open(file_path, "w") as f:
        f.write("0\n0\n" + str(int(sampling_freq)) + "\n")
        for i, val in zip(indices, signal):
            f.write(f"{i} {val:g}\n")
    messagebox.showinfo("Success", f"Signal saved to {file_path}")

def save_quantization_to_txt():
    if quant_results["quantized_signal"] is None:
        messagebox.showwarning("No Data", "Please perform a quantization first.")
        return
    
    dialog = tk.Toplevel()
    dialog.title("Choose Save Format")
    dialog.geometry("400x150")
    dialog.configure(bg="#2c2c2c")
    tk.Label(dialog, text="Select the format for the output file:", bg="#2c2c2c", fg="#f0f0f0", font=('Segoe UI', 11)).pack(pady=15)
    
    def on_select(format_type):
        dialog.destroy()
        _initiate_save(format_type)

    btn_frame = tk.Frame(dialog, bg="#2c2c2c")
    btn_frame.pack(pady=10)
    
    btn_kwargs = {'master': btn_frame, 'width': 20, 'borderwidth': 0, 'font': ('Segoe UI', 9), 'background': '#3c3c3c', 'foreground': '#f0f0f0', 'hoverbackground': '#007bff', 'activebackground': '#0056b3'}
    AnimatedButton(text="encoded and quantized", command=lambda: on_select("test1"), **btn_kwargs).pack(side="left", padx=10)
    AnimatedButton(text="interval index, encoded, quantized and error", command=lambda: on_select("test2"), **btn_kwargs).pack(side="left", padx=10)

def _initiate_save(format_type):
    file_path = filedialog.asksaveasfilename(defaultextension=".txt", title=f"Save as {format_type}")
    if not file_path: return
    data = quant_results
    try:
        with open(file_path, "w") as f:
            f.write(f"0\n0\n{len(data['quantized_signal'])}\n")
            if format_type == "test1":
                for enc, q_val in zip(data['encoded_signal'], data['quantized_signal']):
                    f.write(f"{enc} {q_val:.2f}\n")
            else:
                for inter, enc, q_val, err in zip(data['interval_indices'], data['encoded_signal'], data['quantized_signal'], data['error_signal']):
                    f.write(f"{inter} {enc} {q_val:.3f} {err:.3f}\n")
        messagebox.showinfo("Success", f"Results saved to {file_path}")
    except Exception as e:
        messagebox.showerror("Save Error", f"An error occurred: {e}")

# === Signal Operations ===
def subtract_signals():
    messagebox.showinfo("Subtract", "Select the first signal.")
    i1, s1, fs1 = load_signal_from_file_dialog()
    if s1 is None: return
    messagebox.showinfo("Subtract", "Select the second signal.")
    i2, s2, _ = load_signal_from_file_dialog()
    if s2 is None: return
    if not np.array_equal(i1, i2):
        messagebox.showerror("Mismatch Error", "Signals must have the same indices.")
        return
    res = s1 - s2
    plot_discrete(i1, res, "Subtraction Result")
    update_last_result(i1, res, fs1)

def square_signal():
    i, s, fs = load_signal_from_file_dialog()
    if s is None: return
    res = s ** 2
    plot_discrete(i, res, "Squared Signal")
    update_last_result(i, res, fs)

def normalize_signal(choice):
    i, s, fs = load_signal_from_file_dialog()
    if s is None: return
    min_v, max_v = np.min(s), np.max(s)
    if min_v == max_v:
        s_norm = np.zeros_like(s) if choice == "[-1, 1]" else np.ones_like(s) * min_v
    elif choice == "[-1, 1]":
        s_norm = 2 * (s - min_v) / (max_v - min_v) - 1
    else:
        s_norm = (s - min_v) / (max_v - min_v)
    plot_discrete(i, s_norm, f"Normalized Signal to {choice}")
    update_last_result(i, s_norm, fs)

def accumulate_signal():
    i, s, fs = load_signal_from_file_dialog()
    if s is None: return
    res = np.cumsum(s)
    plot_discrete(i, res, "Accumulated Signal")
    update_last_result(i, res, fs)

def add_signals():
    files = filedialog.askopenfilenames(title="Select two or more signals to add")
    if len(files) < 2: return
    try:
        ref_i, ref_s, ref_fs = read_signal_file(files[0])
        res_s = np.copy(ref_s)
        for f in files[1:]:
            curr_i, curr_s, _ = read_signal_file(f)
            if not np.array_equal(ref_i, curr_i):
                messagebox.showerror("Mismatch", f"Signal in {f.split('/')[-1]} has different indices.")
                return
            res_s += curr_s
        plot_discrete(ref_i, res_s, "Addition Result")
        update_last_result(ref_i, res_s, ref_fs)
    except:
        return

def multiply_by_constant(const_entry):
    try:
        const = float(const_entry.get())
        i, s, fs = load_signal_from_file_dialog()
        if s is None: return
        res = const * s
        plot_discrete(i, res, f"Signal Multiplied by {const}")
        update_last_result(i, res, fs)
    except ValueError:
        messagebox.showerror("Invalid Input", "The constant must be a valid number.")

def convolve_signals():
    messagebox.showinfo("Convolution", "Select the first signal.")
    i1, s1, fs1 = load_signal_from_file_dialog()
    if s1 is None: return
    messagebox.showinfo("Convolution", "Select the second signal.")
    i2, s2, _ = load_signal_from_file_dialog()
    if s2 is None: return
    conv_s = np.convolve(s1, s2)
    start_i = i1[0] + i2[0]
    conv_i = np.arange(start_i, start_i + len(conv_s))
    plot_discrete(conv_i, conv_s, "Convolution Result")
    update_last_result(conv_i, conv_s, fs1)

# --- NEW Signal Operations ---
def shift_signal(shift_amount):
    """Shifts the signal by modifying its indices."""
    indices, samples, fs = load_signal_from_file_dialog()
    if samples is None: return
    
    new_indices = indices + shift_amount
    plot_discrete(new_indices, samples, f"Shifted Signal (by {shift_amount})")
    update_last_result(new_indices, samples, fs)

def reverse_signal():
    """Reverses the signal samples and indices."""
    indices, samples, fs = load_signal_from_file_dialog()
    if samples is None: return
    
    new_samples = np.flip(samples)
    new_indices = np.flip(-indices)
    plot_discrete(new_indices, new_samples, "Reversed Signal")
    update_last_result(new_indices, new_samples, fs)

# def downsample_signal(factor):
#     """Downsamples the signal by an integer factor."""
#     indices, samples, fs = load_signal_from_file_dialog()
#     if samples is None: return
#     if factor <= 0:
#         messagebox.showerror("Invalid Factor", "Downsampling factor must be a positive integer.")
#         return

#     new_indices = indices[::factor]
#     new_samples = samples[::factor]
#     new_fs = fs / factor
#     plot_discrete(new_indices, new_samples, f"Downsampled Signal (Factor: {factor})")
#     update_last_result(new_indices, new_samples, new_fs)

# === Plotting Functions ===
def plot_continuous(time, samples, title="Continuous Signal"):
    """Plots a continuous signal using a line plot."""
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#212121')
    ax.set_facecolor('#2c2c2c')
    ax.plot(time, samples, 'c-', linewidth=1.5)
    ax.set_title(title, fontsize=15, color='white')
    ax.set_xlabel("Time (s)", color='white')
    ax.set_ylabel("Amplitude", color='white')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    plt.tight_layout()
    plt.show()

def plot_discrete(indices, samples, title="Discrete Signal"):
    """Plots a discrete signal using a stem plot."""
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#212121')
    ax.set_facecolor('#2c2c2c')
    markerline, stemlines, baseline = ax.stem(indices, samples, linefmt='c-', markerfmt='co', basefmt='r-')
    plt.setp(stemlines, 'linewidth', 1.5)
    plt.setp(markerline, 'markersize', 5)
    ax.plot(indices, samples, 'c-', alpha=0.5)
    ax.set_title(title, fontsize=15, color='white')
    ax.set_xlabel("Index", color='white')
    ax.set_ylabel("Amplitude", color='white')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    plt.tight_layout()
    plt.show()

def update_last_result(indices, samples, fs):
    """Updates global variables to enable the 'Save Last Result' button."""
    global last_result, save_btn
    last_result['indices'] = indices
    last_result['samples'] = samples
    last_result['fs'] = fs
    if save_btn:
        save_btn.config(state=tk.NORMAL)

# === Integrated Testing ===
def run_test(test_function):
    file_name = filedialog.askopenfilename(title="Select Expected Output File")
    if file_name:
        test_function(file_name)

def SignalSamplesAreEqual(TaskName, output_file, your_i, your_s):
    if your_s is None or your_i is None:
        messagebox.showerror("Test Error", "No signal processed to test.")
        return
    exp_i, exp_s, _ = read_signal_file(output_file)
    if exp_s is None: return
    if not np.array_equal(your_i, exp_i):
        messagebox.showinfo("Test Result", f"{TaskName} Test Failed: Signal indices do not match.")
        return
    if not np.allclose(your_s, exp_s, atol=0.01):
        messagebox.showinfo("Test Result", f"{TaskName} Test Failed: Mismatch in sample values.")
        return
    messagebox.showinfo("Test Result", f"{TaskName} Test Passed Successfully!")

def QuantizationTest1(file_name):
    your_enc, your_quant = quant_results["encoded_signal"], quant_results["quantized_signal"]
    if your_enc is None:
        messagebox.showerror("Test Error", "No quantization performed yet.")
        return
    exp_enc, exp_quant = [], []
    with open(file_name, 'r') as f:
        lines = f.readlines()[3:]
        for line in lines:
            if line.strip():
                parts = line.split()
                if len(parts) == 2:
                    exp_enc.append(parts[0])
                    exp_quant.append(float(parts[1]))
    if your_enc != exp_enc:
        messagebox.showinfo("Test Result", "Quantization Test 1 Failed: Encoded values do not match.")
        return
    if not np.allclose(your_quant, exp_quant, atol=0.01):
        messagebox.showinfo("Test Result", "Quantization Test 1 Failed: Quantized values mismatch.")
        return
    messagebox.showinfo("Test Result", "Quantization Test 1 Passed Successfully!")

def QuantizationTest2(file_name):
    data = quant_results
    if data["interval_indices"] is None:
        messagebox.showerror("Test Error", "No quantization performed yet.")
        return
    exp_inter, exp_enc, exp_quant, exp_err = [], [], [], []
    with open(file_name, 'r') as f:
        lines = f.readlines()[3:]
        for line in lines:
            if line.strip():
                parts = line.split()
                if len(parts) == 4:
                    exp_inter.append(int(parts[0]))
                    exp_enc.append(parts[1])
                    exp_quant.append(float(parts[2]))
                    exp_err.append(float(parts[3]))
    
    if len(data['interval_indices']) != len(exp_inter):
        messagebox.showinfo("Test Result", "Quantization Test 2 Failed: Length of interval indices does not match.")
        return
    if not np.array_equal(data['interval_indices'], exp_inter):
        messagebox.showinfo("Test Result", "Quantization Test 2 Failed: Interval indices do not match.")
        return
        
    if len(data['encoded_signal']) != len(exp_enc):
        messagebox.showinfo("Test Result", "Quantization Test 2 Failed: Length of encoded signals does not match.")
        return
    if data['encoded_signal'] != exp_enc:
        messagebox.showinfo("Test Result", "Quantization Test 2 Failed: Encoded signals do not match.")
        return

    if len(data['quantized_signal']) != len(exp_quant):
        messagebox.showinfo("Test Result", "Quantization Test 2 Failed: Length of quantized signals does not match.")
        return
    if not np.allclose(data['quantized_signal'], exp_quant, atol=0.01):
        messagebox.showinfo("Test Result", "Quantization Test 2 Failed: Quantized values do not match.")
        return

    if len(data['error_signal']) != len(exp_err):
        messagebox.showinfo("Test Result", "Quantization Test 2 Failed: Length of error signals does not match.")
        return
    if not np.allclose(data['error_signal'], exp_err, atol=0.01):
        messagebox.showinfo("Test Result", "Quantization Test 2 Failed: Error signals do not match.")
        return

    messagebox.showinfo("Test Result", "Quantization Test 2 Passed Successfully!")


# === GUI ===
def create_gui():
    global save_btn, save_quant_btn
    root = tk.Tk()
    root.title("Signal Processor Pro")
    root.geometry("800x850") # Increased height for new buttons
    root.configure(bg="#212121")
    
    # --- Fonts ---
    title_font = font.Font(family='Segoe UI', size=18, weight='bold')
    label_font = font.Font(family='Segoe UI', size=11)
    button_font = font.Font(family='Segoe UI', size=10, weight='bold')

    # --- Header ---
    header = tk.Frame(root, bg="#2c2c2c", pady=10)
    header.pack(fill='x')
    tk.Label(header, text="🎛️ Unified Signal Processor", bg="#2c2c2c", fg="#007bff", font=title_font).pack()

    # --- Main content ---
    main_frame = tk.Frame(root, bg="#212121")
    main_frame.pack(expand=True, fill='both', padx=20, pady=20)

    notebook = ttk.Notebook(main_frame)
    style = ttk.Style()
    style.theme_create("dark_notebook", parent="alt", settings={
        "TNotebook": {"configure": {"background": "#212121", "borderwidth": 0}},
        "TNotebook.Tab": {
            "configure": {"padding": [15, 7], "background": "#3c3c3c", "foreground": "#f0f0f0", "font": ('Segoe UI', 10, 'bold')},
            "map": {"background": [("selected", "#007bff")], "foreground": [("selected", "white")]}
        }
    })
    style.theme_use("dark_notebook")
    notebook.pack(expand=1, fill="both")

    # Common button kwargs
    btn_kwargs = {'borderwidth': 0, 'font': button_font, 'background': '#3c3c3c', 
                  'foreground': '#f0f0f0', 'hoverbackground': '#007bff', 
                  'activebackground': '#0056b3', 'width': 25, 'pady': 10}

    # === Tab 1: Signal Generation ===
    gen_tab = tk.Frame(notebook, bg="#2c2c2c", padx=20, pady=20)
    notebook.add(gen_tab, text=' Signal Generation ')
    
    params_frame = tk.Frame(gen_tab, bg="#2c2c2c")
    params_frame.pack(pady=20)
    
    entries = {}
    params = ["Amplitude:", "Phase (rad):", "Analog Freq (Hz):", "Sampling Freq (Hz):"]
    for i, param in enumerate(params):
        tk.Label(params_frame, text=param, font=label_font, bg="#2c2c2c", fg="#f0f0f0").grid(row=i, column=0, padx=10, pady=10, sticky='w')
        entry = tk.Entry(params_frame, font=label_font, bg="#3c3c3c", fg="#f0f0f0", insertbackground="white", width=20, borderwidth=2, relief='flat')
        entry.grid(row=i, column=1, padx=10, pady=10)
        entries[param] = entry

    wave_type = tk.StringVar(value="Sine")
    wave_frame = tk.Frame(gen_tab, bg="#2c2c2c")
    wave_frame.pack(pady=20)
    tk.Radiobutton(wave_frame, text="Sine Wave", variable=wave_type, value="Sine", font=label_font, bg="#2c2c2c", fg="#f0f0f0", selectcolor="#212121", activebackground="#2c2c2c", activeforeground="white", highlightthickness=0).pack(side='left', padx=20)
    tk.Radiobutton(wave_frame, text="Cosine Wave", variable=wave_type, value="Cosine", font=label_font, bg="#2c2c2c", fg="#f0f0f0", selectcolor="#212121", activebackground="#2c2c2c", activeforeground="white", highlightthickness=0).pack(side='left', padx=20)

    def on_generate():
        try:
            A = float(entries["Amplitude:"].get())
            theta = float(entries["Phase (rad):"].get())
            f = float(entries["Analog Freq (Hz):"].get())
            fs = float(entries["Sampling Freq (Hz):"].get())
            t, signal = generate_signal(wave_type.get(), A, theta, f, fs)
            if signal is not None:
                signal_data["signal"], signal_data["fs"] = signal, fs
                plot_continuous(t, signal, f"Generated {wave_type.get()} Wave")
        except ValueError:
            messagebox.showerror("Input Error", "Please ensure all fields have valid numeric values.")

    gen_btn_frame = tk.Frame(gen_tab, bg="#2c2c2c")
    gen_btn_frame.pack(pady=30)
    AnimatedButton(gen_btn_frame, text="📈 Generate Signal", command=on_generate, **btn_kwargs).pack(side='left', padx=10)
    AnimatedButton(gen_btn_frame, text="💾 Save Generated Signal", command=lambda: save_signal_to_txt(np.arange(len(signal_data['signal'])), signal_data['signal'], signal_data['fs']), **btn_kwargs).pack(side='left', padx=10)

    # === Tab 2: Signal Processing ===
    proc_tab = tk.Frame(notebook, bg="#2c2c2c", padx=20, pady=20)
    notebook.add(proc_tab, text=' Signal Processing ')
    
    proc_grid = tk.Frame(proc_tab, bg="#2c2c2c")
    proc_grid.pack(pady=10)
    
    # --- Generic Input Dialog ---
    def open_input_dialog(title, prompt, callback):
        dialog = tk.Toplevel(root)
        dialog.title(title)
        dialog.geometry("300x150")
        dialog.configure(bg="#2c2c2c")
        
        tk.Label(dialog, text=prompt, font=label_font, bg="#2c2c2c", fg="#f0f0f0").pack(pady=10)
        
        input_var = tk.StringVar()
        entry = tk.Entry(dialog, textvariable=input_var, font=label_font, bg="#3c3c3c", fg="#f0f0f0", insertbackground="white", width=15, borderwidth=2, relief='flat')
        entry.pack(pady=5)
        
        def on_proceed():
            try:
                value = int(input_var.get())
                dialog.destroy()
                callback(value)
            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter a valid integer.", parent=dialog)
        
        proceed_btn_kwargs = {'master': dialog, 'borderwidth': 0, 'font': button_font, 'background': '#3c3c3c', 'foreground': '#f0f0f0', 'hoverbackground': '#007bff', 'activebackground': '#0056b3', 'width': 15, 'pady': 8}
        AnimatedButton(text="Proceed", command=on_proceed, **proceed_btn_kwargs).pack(pady=10)
    
    # Quantization
    def open_quantization_dialog():
        dialog = tk.Toplevel(root)
        dialog.title("Configure Quantization")
        dialog.geometry("350x200")
        dialog.configure(bg="#2c2c2c")
        dialog.resizable(False, False)

        bits_var = tk.StringVar()
        levels_var = tk.StringVar()
        
        def update_from_bits(*args):
            try:
                b = int(bits_var.get())
                if b > 0: levels_var.set(str(2**b))
            except (ValueError, tk.TclError): pass
        def update_from_levels(*args):
            try:
                l = int(levels_var.get())
                if l > 1: bits_var.set(str(int(np.ceil(np.log2(l)))))
            except (ValueError, tk.TclError): pass

        bits_var.trace_add("write", update_from_bits)
        levels_var.trace_add("write", update_from_levels)
        
        tk.Label(dialog, text="Number of Bits:", font=label_font, bg="#2c2c2c", fg="#f0f0f0").grid(row=0, column=0, padx=10, pady=15, sticky='w')
        tk.Entry(dialog, textvariable=bits_var, font=label_font, bg="#3c3c3c", fg="#f0f0f0", insertbackground="white", width=15, borderwidth=2, relief='flat').grid(row=0, column=1, padx=10, pady=15)
        
        tk.Label(dialog, text="Number of Levels:", font=label_font, bg="#2c2c2c", fg="#f0f0f0").grid(row=1, column=0, padx=10, pady=15, sticky='w')
        tk.Entry(dialog, textvariable=levels_var, font=label_font, bg="#3c3c3c", fg="#f0f0f0", insertbackground="white", width=15, borderwidth=2, relief='flat').grid(row=1, column=1, padx=10, pady=15)
        
        def on_proceed():
            messagebox.showinfo("Input Signal", "Please select the signal file to quantize.", parent=dialog)
            indices, samples, fs = load_signal_from_file_dialog()
            if samples is None: 
                dialog.destroy()
                return
            
            try:
                if bits_var.get():
                    b = int(bits_var.get())
                    perform_quantization(samples, fs, bits=b)
                elif levels_var.get():
                    l = int(levels_var.get())
                    perform_quantization(samples, fs, levels=l)
                else:
                    messagebox.showerror("Input Error", "Please provide a value for bits or levels.", parent=dialog)
                    return
                dialog.destroy()
            except ValueError:
                messagebox.showerror("Input Error", "Please provide a valid integer.", parent=dialog)
        
        proceed_btn_kwargs = {'master': dialog, 'borderwidth': 0, 'font': button_font, 'background': '#3c3c3c', 'foreground': '#f0f0f0', 'hoverbackground': '#007bff', 'activebackground': '#0056b3', 'width': 15, 'pady': 8}
        AnimatedButton(text="Proceed", command=on_proceed, **proceed_btn_kwargs).grid(row=2, columnspan=2, pady=20)
    
    AnimatedButton(proc_grid, text="Quantize Signal", command=open_quantization_dialog, **btn_kwargs).grid(row=0, column=0, padx=10, pady=5)
    save_quant_btn = AnimatedButton(proc_grid, text="Save Quantization Result", command=save_quantization_to_txt, state=tk.DISABLED, **btn_kwargs)
    save_quant_btn.grid(row=0, column=1, padx=10, pady=5)
    
    # Arithmetic Ops
    AnimatedButton(proc_grid, text="Add Signals", command=add_signals, **btn_kwargs).grid(row=1, column=0, padx=10, pady=5)
    AnimatedButton(proc_grid, text="Subtract Signals", command=subtract_signals, **btn_kwargs).grid(row=1, column=1, padx=10, pady=5)
    AnimatedButton(proc_grid, text="Square Signal", command=square_signal, **btn_kwargs).grid(row=2, column=0, padx=10, pady=5)
    AnimatedButton(proc_grid, text="Accumulate Signal", command=accumulate_signal, **btn_kwargs).grid(row=2, column=1, padx=10, pady=5)
    AnimatedButton(proc_grid, text="Normalize [-1, 1]", command=lambda: normalize_signal('[-1, 1]'), **btn_kwargs).grid(row=3, column=0, padx=10, pady=5)
    AnimatedButton(proc_grid, text="Normalize [0, 1]", command=lambda: normalize_signal('[0, 1]'), **btn_kwargs).grid(row=3, column=1, padx=10, pady=5)

    # New Operations
    AnimatedButton(proc_grid, text="Shift Signal", command=lambda: open_input_dialog("Shift Signal", "Enter Shift Amount:", shift_signal), **btn_kwargs).grid(row=4, column=0, padx=10, pady=5)
    AnimatedButton(proc_grid, text="Reverse Signal", command=reverse_signal, **btn_kwargs).grid(row=4, column=1, padx=10, pady=5)
    # AnimatedButton(proc_grid, text="Downsample Signal", command=lambda: open_input_dialog("Downsample", "Enter Decimation Factor:", downsample_signal), **btn_kwargs).grid(row=5, column=0, padx=10, pady=5)
    
    mult_frame = tk.Frame(proc_tab, bg="#2c2c2c")
    mult_frame.pack(pady=15)
    tk.Label(mult_frame, text="Constant:", font=label_font, bg="#2c2c2c", fg="#f0f0f0").pack(side='left', padx=10)
    const_entry = tk.Entry(mult_frame, font=label_font, bg="#3c3c3c", fg="#f0f0f0", insertbackground="white", width=10, borderwidth=2, relief='flat')
    const_entry.pack(side='left', padx=10)
    AnimatedButton(mult_frame, text="Multiply Signal", command=lambda: multiply_by_constant(const_entry), **btn_kwargs).pack(side='left', padx=10)


    # # === Tab 3: Convolution ===
    # conv_tab = tk.Frame(notebook, bg="#2c2c2c")
    # notebook.add(conv_tab, text=' Convolution ')
    # conv_frame = tk.Frame(conv_tab, bg="#2c2c2c")
    # conv_frame.pack(expand=True)
    # AnimatedButton(conv_frame, text="Perform Convolution", command=convolve_signals, **btn_kwargs).pack()

    # === Tab 4: Testing ===
    test_tab = tk.Frame(notebook, bg="#2c2c2c")
    notebook.add(test_tab, text=' Testing ')
    
    test_frame = tk.Frame(test_tab, bg="#2c2c2c")
    test_frame.pack(pady=20, padx=20)

    tk.Label(test_frame, text="Run Automated Tests", font=('Segoe UI', 14, 'bold'), bg="#2c2c2c", fg="white").pack(pady=(0, 20))

    def run_generic_test(task_name):
        run_test(lambda file_name: SignalSamplesAreEqual(task_name, file_name, last_result['indices'], last_result['samples']))
    
    test_grid = tk.Frame(test_frame, bg="#2c2c2c")
    test_grid.pack()
    
    # Test buttons
    AnimatedButton(test_grid, text="Test Addition", command=lambda: run_generic_test("Addition"), **btn_kwargs).grid(row=0, column=0, padx=10, pady=5)
    AnimatedButton(test_grid, text="Test Subtraction", command=lambda: run_generic_test("Subtraction"), **btn_kwargs).grid(row=0, column=1, padx=10, pady=5)
    AnimatedButton(test_grid, text="Test Squaring", command=lambda: run_generic_test("Squaring"), **btn_kwargs).grid(row=1, column=0, padx=10, pady=5)
    AnimatedButton(test_grid, text="Test Normalization", command=lambda: run_generic_test("Normalization"), **btn_kwargs).grid(row=1, column=1, padx=10, pady=5)
    AnimatedButton(test_grid, text="Test Accumulation", command=lambda: run_generic_test("Accumulation"), **btn_kwargs).grid(row=2, column=0, padx=10, pady=5)
    AnimatedButton(test_grid, text="Test Multiplication", command=lambda: run_generic_test("Multiplication"), **btn_kwargs).grid(row=2, column=1, padx=10, pady=5)
    # AnimatedButton(test_grid, text="Test Convolution", command=lambda: run_generic_test("Convolution"), **btn_kwargs).grid(row=3, column=0, padx=10, pady=5)
    AnimatedButton(test_grid, text="Run Quantization Test 1", command=lambda: run_test(QuantizationTest1), **btn_kwargs).grid(row=4, column=0, padx=10, pady=5)
    AnimatedButton(test_grid, text="Run Quantization Test 2", command=lambda: run_test(QuantizationTest2), **btn_kwargs).grid(row=4, column=1, padx=10, pady=5)
    # New Test Buttons
    AnimatedButton(test_grid, text="Test Shifting", command=lambda: run_generic_test("Shifting"), **btn_kwargs).grid(row=3, column=1, padx=10, pady=5)
    AnimatedButton(test_grid, text="Test Reversing", command=lambda: run_generic_test("Reversing"), **btn_kwargs).grid(row=3, column=0, padx=10, pady=5)
    # AnimatedButton(test_grid, text="Test Downsampling", command=lambda: run_generic_test("Downsampling"), **btn_kwargs).grid(row=5, column=1, padx=10, pady=5)


    # --- Global Save Button ---
    save_btn = AnimatedButton(root, text="💾 Save Last Processed Result", state=tk.DISABLED, **btn_kwargs)
    save_btn.pack(pady=20)
    save_btn.config(command=lambda: save_signal_to_txt(last_result['indices'], last_result['samples'], last_result['fs']), width=30)

    root.mainloop()

if __name__ == "__main__":
    create_gui()
