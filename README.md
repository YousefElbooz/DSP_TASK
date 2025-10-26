## 🧠 OVERVIEW

Your application is a **modular signal processing tool** built with:
- **Tkinter** for GUI
- **NumPy** for signal math
- **Matplotlib** for plotting
- **Pandas** for reading structured signal files

It supports:
- Signal generation (sine/cosine)
- Arithmetic operations (add, subtract, multiply, square, normalize, accumulate)
- Visualization (continuous and discrete)
- Export to `.txt` in a structured format

---

## 🧩 GLOBAL VARIABLES

```python
last_indices, last_samples
```
- Stores the most recently plotted discrete signal for saving.

```python
signal_data = {"signal": None, "fs": None}
```
- Stores the last generated signal and its sampling frequency.

```python
wave_type, const_entry, save_btn
```
- GUI elements used across functions.

---

## 🔧 SIGNAL GENERATION

### `generate_signal(wave_type, amplitude, phase_rad, analog_freq, sampling_freq, duration)`
- **Purpose:** Generate a sine or cosine wave.
- **Logic:**
  - Converts analog frequency to angular frequency.
  - Applies phase shift.
  - Uses `np.linspace` to generate time samples.
  - Computes signal using `np.sin` or `np.cos`.

---

## 📁 FILE OPERATIONS

### `save_signal_to_txt(signal=None, sampling_freq=None)`
- **Purpose:** Save a signal to `.txt` in the format:
  ```
  0
  0
  <sampling_freq>
  <index> <value>
  ```
- **Fallback:** Uses `signal_data` if no signal is passed.

### `load_signal_from_txt()`
- **Purpose:** Load a signal from `.txt` file.
- **Returns:** NumPy array of samples and sampling frequency.

### `read_signal_file(file_path)`
- **Purpose:** Reads signal file using Pandas for discrete operations.
- **Returns:** Indices and samples as NumPy arrays.

---

## ➕ ARITHMETIC OPERATIONS

### `subtract_signals()`
- Loads two signals.
- Checks length and sampling match.
- Computes absolute difference.
- Plots result and enables save.

### `square_signal()`
- Loads one signal.
- Squares each sample.
- Plots and enables save.

### `normalize_signal()`
- Loads one signal.
- Opens a sub-window for user to choose range:
  - `[0, 1]`: min-max normalization
  - `[-1, 1]`: scaled and shifted
- Plots and enables save.

### `accumulate_signal()`
- Loads one signal.
- Computes cumulative sum using `np.cumsum`.
- Plots and enables save.

---

## ✖️ MULTIPLICATION

### `multiply_signal()`
- Loads one signal.
- Multiplies by constant from `const_entry`.
- Plots result using discrete stem plot.

---

## ➕ ADDITION

### `add_signals()`
- Loads multiple signals.
- Checks for length match.
- Adds them element-wise.
- Plots using discrete stem plot.
- Enables save.

---

## 📊 PLOTTING

### `plot_signal(t, signal, title)`
- Plots time-domain signal (continuous).
- Uses `plt.plot`.

### `plot_discrete(indices, samples, title)`
- Plots both continuous and discrete (stem) views.
- Uses `plt.plot` + `plt.stem`.
- Stores `last_indices` and `last_samples` for saving.

---

## 🖥️ GUI STRUCTURE

### `create_gui()`
- Builds the full interface:
  - **Signal Generation Frame**
    - Inputs: amplitude, phase, analog freq, sampling freq
    - Buttons: sine/cosine selection, generate, save
  - **Arithmetic Frame**
    - Buttons: add, multiply
    - Entry: constant
  - **Menu Bar**
    - Arithmetic Ops: subtraction, squaring, normalization, accumulation
  - **Save Button**
    - Disabled by default, enabled after plotting

---

## 🧠 BONUS FUNCTION

### `custom_cumsum(signal)`
- Manual implementation of cumulative sum (alternative to `np.cumsum`)
- Demonstrates algorithmic understanding

---

## ✅ STRENGTHS OF YOUR DESIGN

- **Modular**: Each operation is isolated and reusable
- **User-friendly**: Clear GUI layout with error handling
- **Flexible**: Works with generated or loaded signals
- **Exportable**: Saves results in a structured format
- **Visual**: Time-domain and discrete plots for clarity
