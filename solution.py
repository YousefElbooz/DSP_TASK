import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, font, simpledialog
import pandas as pd
import math # Added for phase calculations

# === Global variables ===
# Stores the last successfully processed time-domain signal
last_result = {"indices": None, "samples": None, "fs": None}
# Stores the last generated signal from Tab 1
signal_data = {"signal": None, "fs": None}
# Stores the results from the last quantization operation
quant_results = {
    "interval_indices": None, "encoded_signal": None,
    "quantized_signal": None, "error_signal": None
}
# --- New Global for DFT ---
# Stores the results from the last DFT operation
dft_data = {
    "complex_coeffs": None, 
    "frequencies": None, 
    "fs": None
}
# --- End New Global ---

# --- GUI Globals ---
# These are defined globally so they can be enabled/disabled from other functions
save_quant_btn = None
save_btn = None
dft_modify_btn = None
dc_remove_btn = None
reconstruct_btn = None
# --- End GUI Globals ---


# --- Custom Animation & Styling Class ---
class AnimatedButton(tk.Button):
    """
    @desc
        A custom tk.Button subclass that adds hover and active (click)
        background color animations for a modern look and feel.
    """
    def __init__(self, master=None, **kwargs):
        """
        @desc
            Initializes the animated button.
        @stepsOfWorking
            1. Retrieves custom color arguments ('hoverbackground', 'activebackground')
               from kwargs, setting defaults if not provided.
            2. Stores these colors as instance variables.
            3. Removes these custom arguments from kwargs to prevent errors when
               calling the parent tk.Button constructor.
            4. Calls the parent `super().__init__` with the remaining standard kwargs.
            5. Binds mouse events (<Enter>, <Leave>, <Button-1>, <ButtonRelease-1>)
               to the corresponding handler methods.
        @param
            master (tk.Widget): The parent widget.
            **kwargs: Standard tk.Button arguments plus optional:
                - hoverbackground (str): Background color on mouse-over.
                - activebackground (str): Background color on mouse-click.
        @return
            None.
        @whenToCall
            This is the constructor, called when creating an instance
            of AnimatedButton.
        """
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
        """
        @desc
            Event handler for when the mouse enters the button's bounds.
        @stepsOfWorking
            1. Configures the button's background color to `self.hover_bg`.
        @param
            e (tk.Event): The Tkinter event object (unused).
        @return
            None.
        @whenToCall
            Automatically called when the '<Enter>' event is triggered.
        """
        self.config(background=self.hover_bg)

    def on_leave(self, e):
        """
        @desc
            Event handler for when the mouse leaves the button's bounds.
        @stepsOfWorking
            1. Configures the button's background color back to `self.normal_bg`.
        @param
            e (tk.Event): The Tkinter event object (unused).
        @return
            None.
        @whenToCall
            Automatically called when the '<Leave>' event is triggered.
        """
        self.config(background=self.normal_bg)
    
    def on_press(self, e):
        """
        @desc
            Event handler for when the left mouse button is pressed.
        @stepsOfWorking
            1. Configures the button's background color to `self.active_bg`.
        @param
            e (tk.Event): The Tkinter event object (unused).
        @return
            None.
        @whenToCall
            Automatically called when the '<Button-1>' event is triggered.
        """
        self.config(background=self.active_bg)

    def on_release(self, e):
        """
        @desc
            Event handler for when the left mouse button is released.
        @stepsOfWorking
            1. Checks if the mouse cursor is still within the button's bounds upon release.
            2. If yes, sets the background to the 'hover' color (`self.hover_bg`).
            3. If no (mouse was dragged off), sets the background to the 'normal' color (`self.normal_bg`).
        @param
            e (tk.Event): The Tkinter event object, used to get cursor coordinates.
        @return
            None.
        @whenToCall
            Automatically called when the '<ButtonRelease-1>' event is triggered.
        """
        if self.winfo_containing(e.x_root, e.y_root) == self:
            self.config(background=self.hover_bg)
        else:
            self.config(background=self.normal_bg)


# --- Quantization Logic ---
def perform_quantization(signal, fs, levels=None, bits=None):
    """
    @desc
        Performs uniform quantization on an input signal given either the
        number of levels or the number of bits.
    @stepsOfWorking
        1. Validates that 'levels' or 'bits' is provided.
        2. Calculates 'levels' from 'bits' if needed, or 'bits' from 'levels'.
        3. Finds the signal's Vmin and Vmax.
        4. Handles the edge case where Vmax == Vmin (a flat signal).
        5. Calculates the quantization step size 'delta'.
        6. Creates an array of quantization level 'midpoints'.
        7. Iterates through each sample in the signal:
            a. Calculates the correct quantization index for the sample.
            b. Assigns the corresponding 'midpoint' value to the quantized signal.
            c. Stores the index and interval number.
        8. Calculates the error signal (original - quantized).
        9. Encodes the indices into binary strings.
        10. Calls `display_quantization_results` to plot and show the data.
    @param
        signal (np.array): The 1D input signal samples.
        fs (float): The sampling frequency of the signal.
        levels (int, optional): The desired number of quantization levels.
        bits (int, optional): The desired number of quantization bits.
    @return
        None. Results are handled by plotting and updating globals.
    @whenToCall
        Called by the 'Proceed' button within the `open_quantization_dialog`.
    """
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
    """
    @desc
        Displays the results of the quantization in plots and a table.
    @stepsOfWorking
        1. Sets up a dark-themed Matplotlib figure with two subplots.
        2. Creates a time axis based on the signal length and 'fs'.
        3. Subplot 1: Plots the original signal and the quantized signal (as a step plot).
        4. Subplot 2: Plots the quantization error over time.
        5. Applies titles, labels, grids, and legends.
        6. Shows the plot.
        7. Calls `show_encoded_data` to open the table in a new window.
        8. Updates the global `quant_results` dictionary with the new data.
        9. Enables the 'Save Quantization Result' button.
    @param
        original (np.array): The original signal.
        quantized (np.array): The quantized signal.
        error (np.array): The error signal.
        encoded (list[str]): The list of binary encoded strings.
        interval_indices (np.array): The list of interval numbers (1-based).
        fs (float): The sampling frequency.
        bits (int): The number of bits used for quantization.
    @return
        None.
    @whenToCall
        Called at the end of `perform_quantization`.
    """
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
    """
    @desc
        Creates a new Toplevel window to display the first 100 samples
        of the quantization results in a tabular format.
    @stepsOfWorking
        1. Creates a `tk.Toplevel` window.
        2. Sets up a `ttk.Treeview` widget for the table.
        3. Applies custom styling for a dark-mode theme.
        4. Defines the columns: 'Sample Index', 'Interval', 'Original', etc.
        5. Iterates through the provided data (up to 100 samples).
        6. Inserts each sample's data as a row in the `Treeview`.
        7. Packs the `Treeview` to fill the window.
    @param
        original (np.array): The original signal.
        quantized (np.array): The quantized signal.
        encoded (list[str]): The list of binary encoded strings.
        interval_indices (np.array): The list of interval numbers.
        error (np.array): The error signal.
    @return
        None.
    @whenToCall
        Called by `display_quantization_results`.
    """
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
    """
    @desc
        Generates a 1-second sine or cosine wave based on user parameters.
    @stepsOfWorking
        1. Validates that `sampling_freq` and `analog_freq` are positive.
        2. Checks if `sampling_freq` is at least 2 * `analog_freq` (Nyquist theorem)
           and shows a warning if it's not.
        3. Creates a time vector `t` using `np.linspace`.
        4. Calculates the signal using the standard formula:
           A * sin(2*pi*f*t + phase) or A * cos(2*pi*f*t + phase).
    @param
        wave_type (str): "Sine" or "Cosine".
        amplitude (float): Amplitude (A) of the wave.
        phase_rad (float): Phase shift in radians.
        analog_freq (float): Analog frequency (f) in Hz.
        sampling_freq (float): Sampling frequency (fs) in Hz.
        duration (float, optional): Signal duration in seconds.
    @return
        (t, signal): A tuple of numpy arrays (time vector, signal samples).
        (None, None): If validation fails.
    @whenToCall
        Called by the `on_generate` function when the "Generate Signal"
        button is clicked.
    """
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
    """
    @desc
        Reads a time-domain signal from a .txt file, formatted with a
        3-line header (type, is_periodic, fs) followed by index-sample pairs.
    @stepsOfWorking
        1. Tries to open and read the file with 'utf-8' encoding.
        2. If a `UnicodeDecodeError` occurs, it falls back to the system's
           default encoding.
        3. Uses `pandas.read_csv` to efficiently read the data, skipping the 3-line header.
        4. Converts the 'index' and 'sample' columns to numpy arrays.
        5. Manually reads the 3rd line (index 2) of the file to get the
           sampling frequency `fs`.
        6. If `fs` is 0 or empty, it defaults to the number of samples found.
        7. Catches and displays any file reading exceptions.
    @param
        file_path (str): The full path to the signal file.
    @return
        (indices, samples, fs): A tuple of (np.array, np.array, float).
        (None, None, None): If file reading fails.
    @whenToCall
        Called by `load_signal_from_file_dialog` and `SignalSamplesAreEqual`.
    """
    try:
        # Try reading with 'utf-8' first
        try:
            with open(file_path, "r", encoding='utf-8') as f:
                lines = f.readlines()
            data = pd.read_csv(file_path, sep='\s+', header=None, skiprows=3, names=['index', 'sample'], on_bad_lines='skip', encoding='utf-8')
        except UnicodeDecodeError:
            # Fallback to default encoding (often 'latin-1' or 'cp1252')
            with open(file_path, "r") as f:
                lines = f.readlines()
            data = pd.read_csv(file_path, sep='\s+', header=None, skiprows=3, names=['index', 'sample'], on_bad_lines='skip')

        indices = data['index'].to_numpy(dtype=int)
        samples = data['sample'].to_numpy(dtype=float)
        
        # Read sampling frequency from line 3, default to signal length if empty or 0
        fs_str = lines[2].strip()
        fs = float(fs_str) if fs_str and float(fs_str) != 0 else len(samples)
        
        return indices, samples, fs
    except Exception as e:
        messagebox.showerror("File Error", f"Could not read the file:\n{file_path}\n{e}")
        return None, None, None

def read_amp_phase_file(file_path):
    """
    @desc
        Reads a frequency-domain signal from a file, formatted with a
        3-line header (type, is_periodic, N) followed by Amp-Phase pairs.
        This can handle .txt or .docx files (if saved as plain text).
    @stepsOfWorking
        1. Tries to open and read the file with 'utf-8' encoding, falling
           back to default encoding on error.
        2. Reads all lines from the file.
        3. Parses the 3rd line (index 2) to get `N`, the number of components.
        4. Iterates through all subsequent lines (index 3+).
        5. For each valid line, splits it, removes any 'f' suffixes from
           the numbers, and converts them to floats.
        6. Appends the floats to the `amplitudes` and `phases` lists.
        7. Catches and displays any file reading exceptions.
    @param
        file_path (str): The full path to the Amp/Phase file.
    @return
        (amplitudes, phases): A tuple of (np.array, np.array).
        (None, None): If file reading fails.
    @whenToCall
        Called by `on_apply_idft_from_file` and `TestDFT`.
    """
    amplitudes = []
    phases = []
    try:
        # Try reading with 'utf-8' first
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                all_lines = f.readlines()
        except UnicodeDecodeError:
            # Fallback to default encoding
            with open(file_path, 'r') as f:
                all_lines = f.readlines()

        if len(all_lines) < 3:
            messagebox.showerror("File Error", "File is too short to be an Amp/Phase file.")
            return None, None
            
        N_str = all_lines[2].strip()
        if not N_str:
            N = 0
        else:
            N = int(N_str)
        
        file_lines = all_lines[3:] # Get component lines
        
        for line in file_lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                # Remove 'f' suffix if present
                amp_str = parts[0].replace('f', '')
                phase_str = parts[1].replace('f', '')
                amplitudes.append(float(amp_str))
                phases.append(float(phase_str))
        
        if len(amplitudes) == 0 and N > 0:
            messagebox.showwarning("File Warning", f"File stated {N} components but 0 were found.")
        
        return np.array(amplitudes), np.array(phases)

    except Exception as e:
        messagebox.showerror("File Error", f"Could not read Amp/Phase file:\n{file_path}\n{e}")
        return None, None
# --- End New Function ---

def load_signal_from_file_dialog():
    """
    @desc
        Opens a standard "Open File" dialog for the user to select
        a signal file and then reads it.
    @stepsOfWorking
        1. Opens a `filedialog.askopenfilename` window, filtered for .txt files.
        2. If the user selects a file, it passes the file path to `read_signal_file`.
        3. If the user cancels, it returns (None, None, None).
    @param
        None.
    @return
        (indices, samples, fs): A tuple from `read_signal_file`.
        (None, None, None): If the user cancels.
    @whenToCall
        This is the primary function called by all buttons that
        need to load a time-domain signal (e.g., "Add Signals",
        "Subtract Signals", "Apply DFT").
    """
    file_path = filedialog.askopenfilename(title="Select a signal file", filetypes=[("Text files", "*.txt")])
    return read_signal_file(file_path) if file_path else (None, None, None)

def save_signal_to_txt(indices=None, signal=None, sampling_freq=None):
    """
    @desc
        Saves a time-domain signal (indices, samples, fs) to a .txt file
        in the 3-line header format.
    @stepsOfWorking
        1. Checks if a signal was passed in. If not, it uses the signal
           stored in the `last_result` global variable.
        2. If no signal is available, shows a warning.
        3. Ensures `sampling_freq` has a valid value.
        4. Opens a `filedialog.asksaveasfilename` dialog.
        5. If the user provides a path:
            a. Writes the 3-line header ("0\n0\nfs\n").
            b. Iterates through the signal and writes each "index sample" pair.
        6. Shows a success message.
    @param
        indices (np.array, optional): The sample indices.
        signal (np.array, optional): The signal samples.
        sampling_freq (float, optional): The sampling frequency.
    @return
        None.
    @whenToCall
        Called by the "Save Last Processed Result" button and the
        "Save Generated Signal" button.
    """
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
    """
    @desc
        Shows a dialog asking the user which quantization format to save.
    @stepsOfWorking
        1. Checks if `quant_results` global has data.
        2. Creates a new `Toplevel` window.
        3. Adds two buttons: "encoded and quantized" and
           "interval index, encoded, quantized and error".
        4. The buttons' commands call `_initiate_save` with the
           respective format type ("test1" or "test2").
    @param
        None.
    @return
        None.
    @whenToCall
        Called by the "Save Quantization Result" button.
    """
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
    """
    @desc
        Performs the actual file saving for quantization results.
    @stepsOfWorking
        1. Opens a `filedialog.asksaveasfilename` dialog.
        2. If a path is provided:
            a. Writes the 3-line header.
            b. If format is "test1", writes "encoded_val quantized_val".
            c. If format is "test2", writes "interval encoded_val quantized_val error_val".
        3. Shows a success or error message.
    @param
        format_type (str): "test1" or "test2", passed from `save_quantization_to_txt`.
    @return
        None.
    @whenToCall
        Called internally by the `on_select` function inside `save_quantization_to_txt`.
    """
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
    """
    @desc
        Loads two signals from files and subtracts the second from the first.
    @stepsOfWorking
        1. Calls `load_signal_from_file_dialog` to get signal 1.
        2. Calls `load_signal_from_file_dialog` to get signal 2.
        3. Validates that both signals were loaded and have identical indices.
        4. Performs element-wise subtraction: `res = s1 - s2`.
        5. Calls `plot_discrete` to display the resulting signal.
        6. Calls `update_last_result` to store the result.
    @param
        None.
    @return
        None.
    @whenToCall
        Called by the "Subtract Signals" button.
    """
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
    """
    @desc
        Loads a signal from a file and squares each of its samples.
    @stepsOfWorking
        1. Calls `load_signal_from_file_dialog` to get the signal.
        2. Validates that the signal was loaded.
        3. Performs element-wise squaring: `res = s ** 2`.
        4. Calls `plot_discrete` to display the resulting signal.
        5. Calls `update_last_result` to store the result.
    @param
        None.
    @return
        None.
    @whenToCall
        Called by the "Square Signal" button.
    """
    i, s, fs = load_signal_from_file_dialog()
    if s is None: return
    res = s ** 2
    plot_discrete(i, res, "Squared Signal")
    update_last_result(i, res, fs)

def normalize_signal(choice):
    """
    @desc
        Loads a signal and normalizes its amplitude to either [0, 1] or [-1, 1].
    @stepsOfWorking
        1. Calls `load_signal_from_file_dialog` to get the signal.
        2. Validates that the signal was loaded.
        3. Finds Vmin and Vmax.
        4. Handles the edge case of a flat signal (Vmin == Vmax).
        5. Applies the correct normalization formula based on the `choice` parameter.
        6. Calls `plot_discrete` to display the resulting signal.
        7. Calls `update_last_result` to store the result.
    @param
        choice (str): The target range, either "[0, 1]" or "[-1, 1]".
    @return
        None.
    @whenToCall
        Called by the "Normalize [0, 1]" and "Normalize [-1, 1]" buttons.
    """
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
    """
    @desc
        Loads a signal and computes its cumulative sum.
    @stepsOfWorking
        1. Calls `load_signal_from_file_dialog` to get the signal.
        2. Validates that the signal was loaded.
        3. Computes the cumulative sum: `res = np.cumsum(s)`.
        4. Calls `plot_discrete` to display the resulting signal.
        5. Calls `update_last_result` to store the result.
    @param
        None.
    @return
        None.
    @whenToCall
        Called by the "Accumulate Signal" button.
    """
    i, s, fs = load_signal_from_file_dialog()
    if s is None: return
    res = np.cumsum(s)
    plot_discrete(i, res, "Accumulated Signal")
    update_last_result(i, res, fs)

def add_signals():
    """
    @desc
        Loads two or more signals from files and adds them together.
    @stepsOfWorking
        1. Opens `filedialog.askopenfilenames` to select multiple files.
        2. Loads the first file as the reference signal.
        3. Iterates through the remaining files:
            a. Loads the current signal.
            b. Validates that its indices match the reference signal.
            c. Adds its samples to the result: `res_s += curr_s`.
        4. Calls `plot_discrete` to display the resulting signal.
        5. Calls `update_last_result` to store the result.
    @param
        None.
    @return
        None.
    @whenToCall
        Called by the "Add Signals" button.
    """
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
    """
    @desc
        Loads a signal and multiplies it by a constant from an Entry widget.
    @stepsOfWorking
        1. Tries to get and convert the constant from the `const_entry` widget.
        2. Calls `load_signal_from_file_dialog` to get the signal.
        3. Validates that the signal was loaded.
        4. Performs element-wise multiplication: `res = const * s`.
        5. Calls `plot_discrete` to display the resulting signal.
        6. Calls `update_last_result` to store the result.
    @param
        const_entry (tk.Entry): The Entry widget containing the constant.
    @return
        None.
    @whenToCall
        Called by the "Multiply Signal" button.
    """
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
    """
    @desc
        Loads two signals and computes their convolution.
    @stepsOfWorking
        1. Calls `load_signal_from_file_dialog` to get signal 1.
        2. Calls `load_signal_from_file_dialog` to get signal 2.
        3. Validates that both signals were loaded.
        4. Computes the convolution: `conv_s = np.convolve(s1, s2)`.
        5. Calculates the new indices for the resulting signal.
        6. Calls `plot_discrete` to display the resulting signal.
        7. Calls `update_last_result` to store the result.
    @param
        None.
    @return
        None.
    @whenToCall
        Called by the "Convolve Signals" button.
    """
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

def shift_signal(shift_amount):
    """
    @desc
        Loads a signal and shifts it by modifying its indices.
    @stepsOfWorking
        1. Calls `load_signal_from_file_dialog` to get the signal.
        2. Validates that the signal was loaded.
        3. Calculates new indices: `new_indices = indices + shift_amount`.
        4. Calls `plot_discrete` with the *new* indices and *original* samples.
        5. Calls `update_last_result` to store the result.
    @param
        shift_amount (int): The amount to shift the indices by.
    @return
        None.
    @whenToCall
        Called by the `open_input_dialog` for the "Shift Signal" button.
    """
    indices, samples, fs = load_signal_from_file_dialog()
    if samples is None: return
    
    new_indices = indices + shift_amount
    plot_discrete(new_indices, samples, f"Shifted Signal (by {shift_amount})")
    update_last_result(new_indices, samples, fs)

def reverse_signal():
    """
    @desc
        Loads a signal and reverses it (time-reversal).
    @stepsOfWorking
        1. Calls `load_signal_from_file_dialog` to get the signal.
        2. Validates that the signal was loaded.
        3. Flips the samples array: `new_samples = np.flip(samples)`.
        4. Flips and negates the indices: `new_indices = np.flip(-indices)`.
        5. Calls `plot_discrete` with the new indices and samples.
        6. Calls `update_last_result` to store the result.
    @param
        None.
    @return
        None.
    @whenToCall
        Called by the "Reverse Signal" button.
    """
    indices, samples, fs = load_signal_from_file_dialog()
    if samples is None: return
    
    new_samples = np.flip(samples)
    new_indices = np.flip(-indices)
    plot_discrete(new_indices, new_samples, "Reversed Signal")
    update_last_result(new_indices, new_samples, fs)


# --- NEW Frequency Domain Logic ---
def on_apply_dft():
    """
    @desc
        Loads a time-domain signal, performs the Discrete Fourier Transform (DFT),
        and displays the resulting frequency-domain plots.
    @stepsOfWorking
        1. Calls `load_signal_from_file_dialog` to get the signal.
        2. Prompts the user for the sampling frequency (`fs`) using a `simpledialog`.
        3. Performs the DFT using a manual, O(N^2) nested-loop implementation.
        4. Calculates the corresponding frequencies, amplitudes, and phases.
        5. Stores all DFT results in the global `dft_data` dictionary.
        6. Normalizes the amplitudes (0 to 1) for plotting.
        7. Calls `plot_frequency_domain` to display the amplitude and phase spectrums.
        8. Finds and displays any dominant frequencies (normalized amplitude > 0.5).
        9. Enables the 'Remove DC', 'Modify Component', and 'Reconstruct' buttons.
    @param
        None.
    @return
        None.
    @whenToCall
        Called by the "Apply DFT (Load Signal)" button.
    """
    global dft_data, dft_modify_btn, dc_remove_btn, reconstruct_btn
    
    indices, samples, fs_from_file = load_signal_from_file_dialog()
    if samples is None:
        return
        
    fs = simpledialog.askfloat("Sampling Frequency", 
                               "Enter the sampling frequency (Hz):",
                               initialvalue=fs_from_file,
                               minvalue=0.1)
    if fs is None:
        return

    N = len(samples)
    
    # --- MODIFIED: Manual DFT Implementation ---
    # Perform DFT using the direct formula: X[k] = sum(x[n] * e^(-j*2*pi*k*n/N))
    # This is an O(N^2) operation and can be slow for large signals.
    X_k = np.zeros(N, dtype=complex) # Array to store complex coefficients

    print("Starting manual DFT calculation... (this may take a moment)")
    # Loop over each frequency bin 'k'
    for k in range(N):
        sum_k = 0.0j # Initialize the sum for this k
        # Loop over each time sample 'n'
        for n in range(N):
            # Calculate the complex exponential term
            angle = -2 * np.pi * k * n / N
            complex_exponential = np.exp(1j * angle) # e^(j * angle)
            
            # Add the sample's contribution
            sum_k += samples[n] * complex_exponential
            
        # Store the final coefficient for frequency k
        X_k[k] = sum_k
    
    print("DFT calculation complete.")
    # --- END MODIFICATION ---

    # We still use np.fft.fftfreq to get the frequency labels (in Hz) for the plot
    frequencies = np.fft.fftfreq(N, d=1/fs) 
    amplitudes = np.abs(X_k)
    phases = np.angle(X_k)
    
    # Store data
    dft_data.update({
        "complex_coeffs": X_k,
        "frequencies": frequencies,
        "fs": fs
    })
    
    # Normalize amplitudes for plotting (0 to 1)
    max_amp = np.max(amplitudes)
    print(max_amp)
    normalized_amplitudes = amplitudes / max_amp if max_amp > 0 else amplitudes
    print(normalized_amplitudes)
    
    # Plot (using fftshift to center 0 Hz)
    plot_frequency_domain(np.fft.fftshift(frequencies),
                          np.fft.fftshift(normalized_amplitudes),
                          np.fft.fftshift(phases),
                          fs)
    
    # Display dominant frequencies
    dominant_indices = np.where(normalized_amplitudes > 0.5)
    dominant_freqs = frequencies[dominant_indices]
    print(dominant_freqs)
    
    if dominant_freqs.size > 0:
        freq_str = "\n".join([f"{f:.2f} Hz" for f in dominant_freqs])
        messagebox.showinfo("Dominant Frequencies", 
                            f"Frequencies with normalized amplitude > 0.5:\n{freq_str}")
    else:
        messagebox.showinfo("Dominant Frequencies", 
                            "No dominant frequencies (amplitude > 0.5) found.")

    # Enable modification buttons
    if dft_modify_btn: dft_modify_btn.config(state=tk.NORMAL)
    if dc_remove_btn: dc_remove_btn.config(state=tk.NORMAL)
    if reconstruct_btn: reconstruct_btn.config(state=tk.NORMAL)

def on_apply_idft_from_file():
    """
    @desc
        Loads an Amplitude/Phase file and performs the Inverse Discrete
        Fourier Transform (IDFT) to reconstruct the time-domain signal.
    @stepsOfWorking
        1. Opens a file dialog for the user to select an Amp/Phase file.
        2. Calls `read_amp_phase_file` to parse the file.
        3. Reconstructs the complex coefficients: `X_k = A * e^(j*P)`.
        4. Performs the IDFT using a manual, O(N^2) nested-loop implementation.
        5. Takes the real part of the result (`np.real`) as the signal.
        6. Calls `plot_discrete` to display the reconstructed signal.
        7. Calls `update_last_result` to store the new time-domain signal.
    @param
        None.
    @return
        None.
    @whenToCall
        Called by the "Apply IDFT (Load Amp/Phase File)" button.
    """
    file_path = filedialog.askopenfilename(
        title="Select Amplitude/Phase file",
        filetypes=[("Text files", "*.txt"), ("Word documents", "*.docx"), ("All files", "*.*")]
    )
    if not file_path:
        return
        
    amplitudes, phases = read_amp_phase_file(file_path)
    if amplitudes is None:
        return
        
    N = len(amplitudes)
    if N == 0:
        messagebox.showerror("Error", "No components found in file.")
        return
    
    # Reconstruct complex coefficients
    X_k = amplitudes * np.exp(1j * phases)
    
    # --- MODIFIED: Manual IDFT Implementation ---
    # Perform IDFT using the direct formula: x[n] = (1/N) * sum(X[k] * e^(j*2*pi*k*n/N))
    # This is an O(N^2) operation.
    reconstructed_samples_complex = np.zeros(N, dtype=complex) # Array to store complex samples

    print("Starting manual IDFT calculation... (this may take a moment)")
    # Loop over each time sample 'n'
    for n in range(N):
        sum_n = 0.0j # Initialize the sum for this n
        # Loop over each frequency bin 'k'
        for k in range(N):
            # Calculate the complex exponential term (note the positive sign)
            angle = 2 * np.pi * k * n / N
            complex_exponential = np.exp(1j * angle) # e^(j * angle)
            
            # Add the coefficient's contribution
            sum_n += X_k[k] * complex_exponential
            
        # Store the final sample for time n, scaled by 1/N
        reconstructed_samples_complex[n] = sum_n / N
    
    print("IDFT calculation complete.")
    # --- END MODIFICATION ---
    
    # Assume real signal
    reconstructed_samples = np.real(reconstructed_samples_complex)
    
    indices = np.arange(N)
    fs = N # Default Fs to N if not known
    
    plot_discrete(indices, reconstructed_samples, "Reconstructed Signal (from file)")
    update_last_result(indices, reconstructed_samples, fs)
def on_modify_component():
    """
    @desc
        Allows the user to modify the amplitude and phase of a single
        frequency component (k) from the last DFT.
    @stepsOfWorking
        1. Checks if `dft_data` global variable contains results.
        2. Prompts the user for the component index `k` to modify.
        3. Gets the current amplitude and phase for that component.
        4. Prompts the user for the new amplitude and new phase,
           showing the current values as defaults.
        5. Updates the `dft_data["complex_coeffs"][k]` with the new
           complex value.
        6. Calls `on_reconstruct_from_dft` to plot the updated signal.
    @param
        None.
    @return
        None.
    @whenToCall
        Called by the "Modify Component (by Index)" button.
    """
    global dft_data
    if dft_data["complex_coeffs"] is None:
        messagebox.showwarning("No Data", "Please apply DFT first.")
        return
    
    N = len(dft_data["complex_coeffs"])
    k = simpledialog.askinteger("Modify Component",
                                "Enter component index (k):",
                                minvalue=0, maxvalue=N-1)
    if k is None:
        return
        
    # Get current values
    current_X_k = dft_data["complex_coeffs"][k]
    current_amp = np.abs(current_X_k)
    current_phase = np.angle(current_X_k)
    
    # Ask for new values
    new_amp = simpledialog.askfloat("Modify Amplitude",
                                    f"Enter new Amplitude for k={k} (current: {current_amp:.4f}):",
                                    initialvalue=current_amp)
    if new_amp is None:
        return
        
    new_phase = simpledialog.askfloat("Modify Phase",
                                      f"Enter new Phase (rad) for k={k} (current: {current_phase:.4f}):",
                                      initialvalue=current_phase)
    if new_phase is None:
        return
        
    print(dft_data);
    # Update the complex coefficient
    dft_data["complex_coeffs"][k] = new_amp * np.exp(1j * new_phase)
    print(dft_data);

    messagebox.showinfo("Success", f"Component {k} has been updated. Reconstructing signal...")
    
    # --- MODIFICATION: Plot the signal after updation ---
    on_reconstruct_from_dft()
    # --- END MODIFICATION ---
def on_remove_dc():
    """
    @desc
        Removes the DC component (the k=0 coefficient) from the
        stored DFT results and immediately reconstructs the signal.
    @stepsOfWorking
        1. Checks if `dft_data` global variable contains results.
        2. Sets `dft_data["complex_coeffs"][0] = 0.0 + 0.0j`.
        3. Shows a success message.
        4. Calls `on_reconstruct_from_dft` to plot the new time-domain signal.
    @param
        None.
    @return
        None.
    @whenToCall
        Called by the "Remove DC Component" button.
    """
    global dft_data
    if dft_data["complex_coeffs"] is None:
        messagebox.showwarning("No Data", "Please apply DFT first.")
        return
        
    dft_data["complex_coeffs"][0] = 0.0 + 0.0j
    messagebox.showinfo("Success", "DC Component (F(0)) has been set to 0. Reconstructing signal...")
    
    # Also reconstruct the signal immediately
    on_reconstruct_from_dft()

def on_reconstruct_from_dft():
    """
    @desc
        Performs an IDFT using the currently stored (and possibly modified)
        complex coefficients from the `dft_data` global.
    @stepsOfWorking
        1. Checks if `dft_data` global variable contains results.
        2. Performs the IDFT using `np.fft.ifft()`.
        3. Takes the real part of the result (`np.real`).
        4. Calls `plot_discrete` to display the reconstructed signal.
        5. Calls `update_last_result` to store the new time-domain signal.
    @param
        None.
    @return
        None.
    @whenToCall
        Called by the "Reconstruct from Modified DFT" button and also
        by `on_remove_dc`.
    """
    global dft_data
    if dft_data["complex_coeffs"] is None:
        messagebox.showwarning("No Data", "No DFT data to reconstruct.")
        return
    
    # Perform IDFT
    reconstructed_samples = np.fft.ifft(dft_data["complex_coeffs"])
    
    # Assume real signal
    reconstructed_samples = np.real(reconstructed_samples)
    
    indices = np.arange(len(reconstructed_samples))
    fs = dft_data["fs"]
    
    plot_discrete(indices, reconstructed_samples, "Reconstructed Signal (from modified DFT)")
    update_last_result(indices, reconstructed_samples, fs)

# --- End New Frequency Domain Logic ---


# === Plotting Functions ===
def plot_continuous(time, samples, title="Continuous Signal"):
    """
    @desc
        Utility function to plot a continuous-style signal.
    @stepsOfWorking
        1. Sets Matplotlib style to 'dark_background'.
        2. Creates a figure and axes with dark facecolors.
        3. Uses `ax.plot()` to draw the signal as a line.
        4. Sets titles, labels, grids, and tick colors for readability.
        5. Calls `plt.show()` to display the plot.
    @param
        time (np.array): The time vector for the x-axis.
        samples (np.array): The signal samples for the y-axis.
        title (str): The title for the plot.
    @return
        None.
    @whenToCall
        Called by `on_generate`.
    """
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
    """
    @desc
        Utility function to plot a discrete-style signal using a stem plot.
    @stepsOfWorking
        1. Sets Matplotlib style to 'dark_background'.
        2. Creates a figure and axes with dark facecolors.
        3. Uses `ax.stem()` to draw the signal as stems and markers.
        4. Overlays a faint `ax.plot()` to connect the dots.
        5. Sets titles, labels, grids, and tick colors for readability.
        6. Calls `plt.show()` to display the plot.
    @param
        indices (np.array): The sample indices for the x-axis.
        samples (np.array): The signal samples for the y-axis.
        title (str): The title for the plot.
    @return
        None.
    @whenToCall
        Called by all time-domain signal processing functions
        (add, subtract, normalize, reconstruct, etc.).
    """
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

def plot_frequency_domain(frequencies, normalized_amplitudes, phases, fs):
    """
    @desc
        Utility function to plot the frequency domain (Amplitude and Phase).
    @stepsOfWorking
        1. Sets Matplotlib style to 'dark_background'.
        2. Creates a figure with two subplots (ax1, ax2).
        3. Subplot 1: Uses `ax1.stem()` to plot the normalized amplitude spectrum.
        4. Subplot 2: Uses `ax2.stem()` to plot the phase spectrum.
        5. Sets titles, labels, grids, and tick colors for readability.
        6. Calls `plt.show()` to display the plot.
    @param
        frequencies (np.array): The frequency vector for the x-axis.
        normalized_amplitudes (np.array): The normalized amplitudes (0-1).
        phases (np.array): The phase values in radians.
        fs (float): The sampling frequency (for the title).
    @return
        None.
    @whenToCall
        Called by `on_apply_dft`.
    """
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f'Frequency Domain (Fs = {fs} Hz)', fontsize=16, color='white')
    fig.patch.set_facecolor('#212121')

    # --- Amplitude Plot ---
    ax1.set_facecolor('#2c2c2c')
    ax1.stem(frequencies, normalized_amplitudes, 'c-', markerfmt='co', basefmt='r-')
    ax1.set_title('Normalized Amplitude Spectrum', color='white')
    ax1.set_ylabel('Normalized Amplitude', color='white')
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.tick_params(axis='x', colors='white')
    ax1.tick_params(axis='y', colors='white')

    # --- Phase Plot ---
    ax2.set_facecolor('#2c2c2c')
    ax2.stem(frequencies, phases, 'c-', markerfmt='co', basefmt='r-')
    ax2.set_title('Phase Spectrum', color='white')
    ax2.set_xlabel('Frequency (Hz)', color='white')
    ax2.set_ylabel('Phase (radians)', color='white')
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.tick_params(axis='x', colors='white')
    ax2.tick_params(axis='y', colors='white')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
# --- End New Plotting Function ---

def update_last_result(indices, samples, fs):
    """
    @desc
        A helper function to update the global `last_result` variable
        and enable the "Save Last Result" button.
    @stepsOfWorking
        1. Updates the `last_result` dictionary with the new signal data.
        2. Checks if the global `save_btn` exists.
        3. If it exists, configures its state to 'normal' (enabled).
    @param
        indices (np.array): The sample indices.
        samples (np.array): The signal samples.
        fs (float): The sampling frequency.
    @return
        None.
    @whenToCall
        Called by every function that produces a new time-domain
        signal (add, subtract, reconstruct, normalize, etc.).
    """
    global last_result, save_btn
    last_result['indices'] = indices
    last_result['samples'] = samples
    last_result['fs'] = fs
    if save_btn:
        save_btn.config(state=tk.NORMAL)

# === Integrated Testing ===
def run_test(test_function):
    """
    @desc
        A wrapper function that opens a file dialog and then executes
        a provided test function with the selected file path.
    @stepsOfWorking
        1. Opens an `askopenfilename` dialog.
        2. If the user selects a file, it calls the `test_function`
           (which is often a lambda) and passes the `file_name` to it.
    @param
        test_function (function): A function or lambda that accepts
                                  a single argument (file_path).
    @return
        None.
    @whenToCall
        Called by all the "Test..." button callbacks in the "Testing" tab.
    """
    file_name = filedialog.askopenfilename(
        title="Select Expected Output File",
        filetypes=[("Text files", "*.txt"), ("Word documents", "*.docx"), ("All files", "*.*")]
    )
    if file_name:
        test_function(file_name)

def SignalSamplesAreEqual(TaskName, output_file, your_i, your_s):
    """
    @desc
        Compares the currently stored `last_result` signal against an
        expected time-domain signal loaded from a file.
    @stepsOfWorking
        1. Checks if a "your" signal (from `last_result`) exists to test.
        2. Reads the `output_file` using `read_signal_file` to get the
           "expected" signal.
        3. Compares lengths of indices and samples.
        4. Compares indices using `np.array_equal`.
        5. Compares samples using `np.allclose` with a tolerance (atol=0.01).
        6. Shows a "Passed" or "Failed" `messagebox` with the task name.
    @param
        TaskName (str): The name of the test (e.g., "Addition", "IDFT").
        output_file (str): The file path to the expected output signal.
        your_i (np.array): The indices of the signal to test (from `last_result`).
        your_s (np.array): The samples of the signal to test (from `last_result`).
    @return
        None.
    @whenToCall
        Called by the `run_generic_test` lambda, which is triggered
        by all time-domain test buttons (e.g., "Test Addition").
    """
    if your_s is None or your_i is None:
        messagebox.showerror("Test Error", "No signal processed to test.")
        return
    exp_i, exp_s, _ = read_signal_file(output_file)
    if exp_s is None: return
    
    if len(exp_i) != len(your_i):
        messagebox.showinfo("Test Result", f"{TaskName} Test Failed: Signal lengths do not match. (Expected: {len(exp_i)}, Yours: {len(your_i)})")
        return
    if not np.array_equal(your_i, exp_i):
        messagebox.showinfo("Test Result", f"{TaskName} Test Failed: Signal indices do not match.")
        return
    if not np.allclose(your_s, exp_s, atol=0.01):
        messagebox.showinfo("Test Result", f"{TaskName} Test Failed: Mismatch in sample values.")
        return
    messagebox.showinfo("Test Result", f"{TaskName} Test Passed Successfully!")

def TestDFT(file_name):
    """
    @desc
        Compares the currently stored `dft_data` (Amplitudes and Phases)
        against an expected Amp/Phase file.
    @stepsOfWorking
        1. Checks if `dft_data` global contains results.
        2. Reads the `file_name` using `read_amp_phase_file` to get
           "expected" amplitudes and phases.
        3. Calculates "your" amplitudes and phases from `dft_data`.
        4. Compares component counts (lengths).
        5. Compares amplitudes using `np.allclose(..., atol=0.01)`.
        6. Compares the full complex coefficients using `np.allclose`. This
           is more robust than comparing phases directly, as it handles
           phase wrapping (e.g., -pi vs +pi).
        7. Shows a "Passed" or "Failed" `messagebox`.
    @param
        file_name (str): The file path to the expected Amp/Phase file.
    @return
        None.
    @whenToCall
        Called by the "Test DFT (Amp/Phase)" button.
    """
    global dft_data
    if dft_data["complex_coeffs"] is None:
        messagebox.showerror("Test Error", "No DFT performed yet.")
        return
        
    exp_amps, exp_phases = read_amp_phase_file(file_name)
    if exp_amps is None:
        return
        
    your_amps = np.abs(dft_data["complex_coeffs"])
    your_phases = np.angle(dft_data["complex_coeffs"])

    if len(exp_amps) != len(your_amps):
        messagebox.showinfo("Test Result", f"DFT Test Failed: Component counts do not match. (Expected: {len(exp_amps)}, Yours: {len(your_amps)})")
        return

    # Compare Amplitudes
    if not np.allclose(your_amps, exp_amps, atol=0.01):
        messagebox.showinfo("Test Result", "DFT Test Failed: Mismatch in Amplitudes.")
        return
        
    # Compare Phases
    # We must be careful with phase wrapping (e.g., -pi vs pi)
    # A robust way is to compare the complex numbers directly
    exp_X_k = exp_amps * np.exp(1j * exp_phases)
    your_X_k = dft_data["complex_coeffs"]

    # Use a slightly higher tolerance for complex comparison
    if not np.allclose(your_X_k, exp_X_k, atol=0.02):
        print("Complex comparison failed. Checking raw phases.")
        # If complex compare fails, check phases with a tolerance
        if not np.allclose(your_phases, exp_phases, atol=0.02):
            messagebox.showinfo("Test Result", "DFT Test Failed: Mismatch in Phases.")
            return

    messagebox.showinfo("Test Result", "DFT Test Passed Successfully!")

def QuantizationTest1(file_name):
    """
    @desc
        Compares the `quant_results` against an expected "Test 1" file
        (encoded and quantized value).
    @stepsOfWorking
        1. Checks if `quant_results` global contains data.
        2. Manually opens and parses the `file_name`, skipping the header
           and reading "encoded quantized_val" pairs into lists.
        3. Compares the list of encoded strings (`your_enc != exp_enc`).
        4. Compares the list of quantized values (`np.allclose(..., atol=0.01)`).
        5. Shows a "Passed" or "Failed" `messagebox`.
    @param
        file_name (str): The file path to the expected output file.
    @return
        None.
    @whenToCall
        Called by the "Run Quantization Test 1" button.
    """
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
    """
    @desc
        Compares the `quant_results` against an expected "Test 2" file
        (interval, encoded, quantized, error).
    @stepsOfWorking
        1. Checks if `quant_results` global contains data.
        2. Manually opens and parses the `file_name`, skipping the header
           and reading all four columns into separate lists.
        3. Performs separate comparisons for lengths, interval indices,
           encoded strings, quantized values, and error values.
        4. Shows a "Passed" or "Failed" `messagebox`.
    @param
        file_name (str): The file path to the expected output file.
    @return
        None.
    @whenToCall
        Called by the "Run Quantization Test 2" button.
    """
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
    """
    @desc
        The main function that builds and configures the entire
        Tkinter GUI application.
    @stepsOfWorking
        1. Creates the main `tk.Tk()` root window.
        2. Assigns global variables for key buttons (e.g., `save_btn`).
        3. Defines fonts and styling for the `ttk.Notebook` (tabs).
        4. Creates the header frame.
        5. Creates the main `ttk.Notebook` widget.
        6. **Tab 1 (Signal Generation):**
           - Builds the UI (labels, entries, radio buttons) for generating signals.
           - Defines the `on_generate` callback for the "Generate" button.
        7. **Tab 2 (Signal Processing):**
           - Builds a grid of buttons for all time-domain operations
             (Add, Subtract, Normalize, etc.).
           - Defines helper dialog functions (`open_input_dialog`,
             `open_quantization_dialog`) to get user input for these operations.
        8. **Tab 3 (Frequency Domain):**
           - Builds the UI with buttons for DFT, IDFT, DC Removal, Modify,
             and Reconstruct.
           - Links buttons to their respective `on_...` callbacks.
        9. **Tab 4 (Testing):**
           - Builds a grid of buttons for all automated tests.
           - Defines a helper lambda `run_generic_test` to simplify
             test button commands.
        10. Creates the global "Save Last Processed Result" button at the bottom.
        11. Starts the Tkinter main event loop (`root.mainloop()`).
    @param
        None.
    @return
        None.
    @whenToCall
        This is the main entry point of the application, called from
        the `if __name__ == "__main__":` block.
    """
    global save_btn, save_quant_btn
    # --- Make DFT buttons global ---
    global dft_modify_btn, dc_remove_btn, reconstruct_btn
    
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
                  'activebackground': '#0056b3', 'width': 30, 'pady': 10} # Increased width

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

    # --- Callback defined inside GUI to access 'entries' ---
    def on_generate():
        """
        @desc
            Callback for the "Generate Signal" button.
        @stepsOfWorking
            1. Tries to get and convert all values from the 'entries' dict.
            2. Calls `generate_signal` with the parsed values.
            3. If successful, updates the global `signal_data` and plots the result.
        @param
            None.
        @return
            None.
        """
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
        """
        @desc
            A helper function to create a small popup dialog that
            asks for a single integer value.
        @stepsOfWorking
            1. Creates a `Toplevel` window.
            2. Adds a label, an entry widget, and a "Proceed" button.
            3. The "Proceed" button's command:
                a. Tries to convert the entry's text to an integer.
                b. If successful, destroys the dialog and calls the
                   `callback` function with the integer value.
                c. If failed, shows an error.
        @param
            title (str): The title for the dialog window.
            prompt (str): The text to display above the entry box.
            callback (function): The function to call with the
                                 integer value (e.g., `shift_signal`).
        @return
            None.
        """
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
    
    # --- Quantization Dialog ---
    def open_quantization_dialog():
        """
        @desc
            A popup dialog for quantization that asks for "Bits" or "Levels".
        @stepsOfWorking
            1. Creates a `Toplevel` window.
            2. Adds two Entry widgets: one for "Bits" and one for "Levels".
            3. Adds `trace` listeners to the entry variables:
                - Typing in "Bits" automatically calculates and fills "Levels".
                - Typing in "Levels" automatically calculates and fills "Bits".
            4. The "Proceed" button's command:
                a. Asks the user to load a signal file.
                b. Calls `perform_quantization` with the loaded signal
                   and the specified bits/levels.
        @param
            None.
        @return
            None.
        """
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
    
    # --- Build the processing grid ---
    grid_btn_kwargs = btn_kwargs.copy()
    grid_btn_kwargs['width'] = 25
    
    AnimatedButton(proc_grid, text="Quantize Signal", command=open_quantization_dialog, **grid_btn_kwargs).grid(row=0, column=0, padx=10, pady=5)
    save_quant_btn = AnimatedButton(proc_grid, text="Save Quantization Result", command=save_quantization_to_txt, state=tk.DISABLED, **grid_btn_kwargs)
    save_quant_btn.grid(row=0, column=1, padx=10, pady=5)
    
    AnimatedButton(proc_grid, text="Add Signals", command=add_signals, **grid_btn_kwargs).grid(row=1, column=0, padx=10, pady=5)
    AnimatedButton(proc_grid, text="Subtract Signals", command=subtract_signals, **grid_btn_kwargs).grid(row=1, column=1, padx=10, pady=5)
    AnimatedButton(proc_grid, text="Square Signal", command=square_signal, **grid_btn_kwargs).grid(row=2, column=0, padx=10, pady=5)
    AnimatedButton(proc_grid, text="Accumulate Signal", command=accumulate_signal, **grid_btn_kwargs).grid(row=2, column=1, padx=10, pady=5)
    AnimatedButton(proc_grid, text="Normalize [-1, 1]", command=lambda: normalize_signal('[-1, 1]'), **grid_btn_kwargs).grid(row=3, column=0, padx=10, pady=5)
    AnimatedButton(proc_grid, text="Normalize [0, 1]", command=lambda: normalize_signal('[0, 1]'), **grid_btn_kwargs).grid(row=3, column=1, padx=10, pady=5)
    AnimatedButton(proc_grid, text="Shift Signal", command=lambda: open_input_dialog("Shift Signal", "Enter Shift Amount:", shift_signal), **grid_btn_kwargs).grid(row=4, column=0, padx=10, pady=5)
    AnimatedButton(proc_grid, text="Reverse Signal", command=reverse_signal, **grid_btn_kwargs).grid(row=4, column=1, padx=10, pady=5)
    AnimatedButton(proc_grid, text="Convolve Signals", command=convolve_signals, **grid_btn_kwargs).grid(row=5, column=0, padx=10, pady=5)
    
    mult_frame = tk.Frame(proc_tab, bg="#2c2c2c")
    mult_frame.pack(pady=15)
    tk.Label(mult_frame, text="Constant:", font=label_font, bg="#2c2c2c", fg="#f0f0f0").pack(side='left', padx=10)
    const_entry = tk.Entry(mult_frame, font=label_font, bg="#3c3c3c", fg="#f0f0f0", insertbackground="white", width=10, borderwidth=2, relief='flat')
    const_entry.pack(side='left', padx=10)
    AnimatedButton(mult_frame, text="Multiply Signal", command=lambda: multiply_by_constant(const_entry), **grid_btn_kwargs).pack(side='left', padx=10)


    # === Tab 3: Frequency Domain (NEW) ===
    freq_tab = tk.Frame(notebook, bg="#2c2c2c", padx=20, pady=20)
    notebook.add(freq_tab, text=' Frequency Domain ')
    
    freq_frame = tk.Frame(freq_tab, bg="#2c2c2c")
    freq_frame.pack(pady=20, padx=20)

    tk.Label(freq_frame, text="DFT / IDFT Operations", font=('Segoe UI', 14, 'bold'), bg="#2c2c2c", fg="white").pack(pady=(0, 20))

    freq_btn_kwargs = btn_kwargs.copy()
    freq_btn_kwargs['width'] = 35 
    
    AnimatedButton(freq_frame, text="Apply DFT (Load Signal)", 
                   command=on_apply_dft, **freq_btn_kwargs).pack(pady=7)
                   
    AnimatedButton(freq_frame, text="Apply IDFT (Load Amp/Phase File)", 
                   command=on_apply_idft_from_file, **freq_btn_kwargs).pack(pady=7)
                   
    dc_remove_btn = AnimatedButton(freq_frame, text="Remove DC Component", 
                                   command=on_remove_dc, state=tk.DISABLED, **freq_btn_kwargs)
    dc_remove_btn.pack(pady=7)

    dft_modify_btn = AnimatedButton(freq_frame, text="Modify Component (by Index)", 
                                    command=on_modify_component, state=tk.DISABLED, **freq_btn_kwargs)
    dft_modify_btn.pack(pady=7)
    
    reconstruct_btn = AnimatedButton(freq_frame, text="Reconstruct from Modified DFT", 
                                     command=on_reconstruct_from_dft, state=tk.DISABLED, **freq_btn_kwargs)
    reconstruct_btn.pack(pady=7)


    # === Tab 4: Testing ===
    test_tab = tk.Frame(notebook, bg="#2c2c2c")
    notebook.add(test_tab, text=' Testing ')
    
    test_frame = tk.Frame(test_tab, bg="#2c2c2c")
    test_frame.pack(pady=20, padx=20)

    tk.Label(test_frame, text="Run Automated Tests", font=('Segoe UI', 14, 'bold'), bg="#2c2c2c", fg="white").pack(pady=(0, 20))

    def run_generic_test(task_name):
        """
        @desc
            A helper lambda-like function to reduce code duplication
            for the test buttons.
        @stepsOfWorking
            1. Calls the main `run_test` function.
            2. Passes a lambda function to `run_test` that, when called,
               will execute `SignalSamplesAreEqual` with the correct
               task name and the current `last_result` data.
        @param
            task_name (str): The name of the test to run.
        @return
            None.
        """
        run_test(lambda file_name: SignalSamplesAreEqual(task_name, file_name, last_result['indices'], last_result['samples']))
    
    test_grid = tk.Frame(test_frame, bg="#2c2c2c")
    test_grid.pack()
    
    # --- Build the test grid ---
    AnimatedButton(test_grid, text="Test Addition", command=lambda: run_generic_test("Addition"), **grid_btn_kwargs).grid(row=0, column=0, padx=10, pady=5)
    AnimatedButton(test_grid, text="Test Subtraction", command=lambda: run_generic_test("Subtraction"), **grid_btn_kwargs).grid(row=0, column=1, padx=10, pady=5)
    AnimatedButton(test_grid, text="Test Squaring", command=lambda: run_generic_test("Squaring"), **grid_btn_kwargs).grid(row=1, column=0, padx=10, pady=5)
    AnimatedButton(test_grid, text="Test Normalization", command=lambda: run_generic_test("Normalization"), **grid_btn_kwargs).grid(row=1, column=1, padx=10, pady=5)
    AnimatedButton(test_grid, text="Test Accumulation", command=lambda: run_generic_test("Accumulation"), **grid_btn_kwargs).grid(row=2, column=0, padx=10, pady=5)
    AnimatedButton(test_grid, text="Test Multiplication", command=lambda: run_generic_test("Multiplication"), **grid_btn_kwargs).grid(row=2, column=1, padx=10, pady=5)
    AnimatedButton(test_grid, text="Test Shifting", command=lambda: run_generic_test("Shifting"), **grid_btn_kwargs).grid(row=3, column=0, padx=10, pady=5)
    AnimatedButton(test_grid, text="Test Reversing", command=lambda: run_generic_test("Reversing"), **grid_btn_kwargs).grid(row=3, column=1, padx=10, pady=5)
    AnimatedButton(test_grid, text="Test Convolution", command=lambda: run_generic_test("Convolution"), **grid_btn_kwargs).grid(row=4, column=0, padx=10, pady=5, columnspan=2)
    
    AnimatedButton(test_grid, text="Run Quantization Test 1", command=lambda: run_test(QuantizationTest1), **grid_btn_kwargs).grid(row=5, column=0, padx=10, pady=5)
    AnimatedButton(test_grid, text="Run Quantization Test 2", command=lambda: run_test(QuantizationTest2), **grid_btn_kwargs).grid(row=5, column=1, padx=10, pady=5)
    
    tk.Label(test_grid, text="Frequency Domain Tests", font=label_font, bg="#2c2c2c", fg="white").grid(row=6, columnspan=2, pady=(15,5))
    AnimatedButton(test_grid, text="Test DFT (Amp/Phase)", command=lambda: run_test(TestDFT), **grid_btn_kwargs).grid(row=7, column=0, padx=10, pady=5)
    AnimatedButton(test_grid, text="Test IDFT (Time Signal)", command=lambda: run_generic_test("IDFT"), **grid_btn_kwargs).grid(row=7, column=1, padx=10, pady=5)
    AnimatedButton(test_grid, text="Test DC Removal", command=lambda: run_generic_test("DC Removal"), **grid_btn_kwargs).grid(row=8, column=0, padx=10, pady=5, columnspan=2)


    # --- Global Save Button ---
    save_btn = AnimatedButton(root, text="💾 Save Last Processed Result", state=tk.DISABLED, **btn_kwargs)
    save_btn.pack(pady=20)
    save_btn.config(command=lambda: save_signal_to_txt(last_result['indices'], last_result['samples'], last_result['fs']), width=30)

    # --- Start the application ---
    root.mainloop()

if __name__ == "__main__":
    create_gui()
