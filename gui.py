import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.optimize import root_scalar
from sympy import symbols, solve


class CompMathApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CompMath Final Project")
        self.center_window(self.root)
        self.create_main_menu()
        style = ttk.Style()
        style.configure("TButton",
                        font=("Arial", 12),
                        padding=10,
                        background="#4CAF50",
                        foreground="black")
        style.map("TButton",
                  background=[("active", "#45a049")])

    def center_window(self, window, width=800, height=600):
        window.update_idletasks()
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        window.geometry(f"{width}x{height}+{x}+{y}")

    def create_main_menu(self):
        ttk.Label(self.root, text="CompMath Final!", font=("Arial", 14)).pack(pady=10)
        ttk.Label(self.root, text="Select a task:", font=("Arial", 14)).pack(pady=10)

        tasks = [
            ("Graphical Method & Absolute Error", self.task1),
            ("Comparison of Root-Finding Methods", self.task2_root_finding),
            ("Relaxation Method", self.task3_relaxation),
            ("Power Method for Eigenvalues", self.task4_power_method),
            ("Exponential Curve Fitting", self.task5_curve_fitting),
            ("Cubic Spline Interpolation", self.task6_cubic_spline),
            ("Modified Euler’s Method", self.task7_euler),
            ("Weddle’s Rule Integration", self.task8_weddle)
        ]

        for text, command in tasks:
            ttk.Button(self.root, text=text, command=command, style="TButton").pack(fill='x', padx=20, pady=5)

    def task1(self):
        task_window = tk.Toplevel(self.root)
        task_window.title("Graphical Method & Absolute Error")
        self.center_window(task_window)

        ttk.Label(task_window, text="For f(x) = x^5 - 4x^4 + 6x^3 - 4x + 1 ").pack()
        ttk.Label(task_window, text="Enter x range (start, end):").pack()
        # inputs
        x_start = tk.Entry(task_window, font=("Arial", 12), bg="#f0f0f0", fg="#333", bd=2, relief="solid")
        x_start.pack()
        x_end = tk.Entry(task_window, font=("Arial", 12), bg="#f0f0f0", fg="#333", bd=2, relief="solid")

        x_end.pack()

        result_frame = tk.Frame(task_window)
        result_frame.pack()

        def compute():
            try:
                start = float(x_start.get())
                end = float(x_end.get())
                x = np.linspace(start, end, 100)
                y = x ** 5 - 4 * x ** 4 + 6 * x ** 3 - 4 * x + 1

                # Plot the graph
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(x, y, label="f(x) = x^5 - 4x^4 + 6x^3 - 4x + 1")
                ax.axhline(0, color='black', linewidth=0.5)
                ax.axvline(0, color='black', linewidth=0.5)
                ax.legend()
                ax.set_xlabel("x")
                ax.set_ylabel("f(x)")
                ax.set_title("Graphical Method")
                ax.grid()

                # Draw graph on window
                for widget in result_frame.winfo_children():
                    widget.destroy()

                canvas = FigureCanvasTkAgg(fig, master=result_frame)
                canvas.draw()
                canvas.get_tk_widget().pack()

                # Finding exact root using SymPy
                x_sym = symbols('x')
                f_sym = x_sym ** 5 - 4 * x_sym ** 4 + 6 * x_sym ** 3 - 4 * x_sym + 1
                exact_roots = [float(r.evalf()) for r in solve(f_sym, x_sym) if r.is_real]

                f = lambda x: x ** 5 - 4 * x ** 4 + 6 * x ** 3 - 4 * x + 1
                root_approx = None
                intervals = np.linspace(start, end, 10)

                # Bisection method via root_scalar
                for i in range(len(intervals) - 1):
                    a, b = intervals[i], intervals[i + 1]
                    if f(a) * f(b) < 0:
                        res = root_scalar(f, bracket=[a, b], method='bisect')
                        if res.converged:
                            root_approx = res.root
                            break

                if not exact_roots:
                    result_text = "No exact root found in the given range."
                else:
                    # Outputs
                    exact_root = min(exact_roots, key=lambda r: abs(r - root_approx)) if root_approx else exact_roots[0]
                    abs_error = abs(root_approx - exact_root) if root_approx else "N/A"
                    result_text = f"Approximate Root: {root_approx}\nExact Root: {exact_root}\nAbsolute Error: {abs_error}"

                ttk.Label(result_frame, text=result_text, font=("Arial", 12)).pack()

            except ValueError:
                messagebox.showerror("Input Error", "Please enter valid numbers.")

        # Button for next values of x
        ttk.Button(task_window, text="Plot Graph & Compute Root", command=compute).pack(pady=5)

    def task2_root_finding(self):
        self.create_task_window("Comparison of Root-Finding Methods")

    def task3_relaxation(self):
        self.create_task_window("Relaxation Method")

    def task4_power_method(self):
        self.create_task_window("Power Method for Eigenvalues")

    def task5_curve_fitting(self):
        self.create_task_window("Exponential Curve Fitting")

    def task6_cubic_spline(self):
        self.create_task_window("Cubic Spline Interpolation")

    def task7_euler(self):
        self.create_task_window("Modified Euler’s Method")

    def task8_weddle(self):
        self.create_task_window("Weddle’s Rule Integration")

    def create_task_window(self, title):
        task_window = tk.Toplevel(self.root)
        task_window.title(title)
        ttk.Label(task_window, text=f"{title}", font=("Arial", 12)).pack(pady=10)
        ttk.Button(task_window, text="Compute", command=lambda: messagebox.showinfo("Result", "Computation Pending")) \
            .pack(pady=5)


if __name__ == "__main__":
    root = tk.Tk()
    app = CompMathApp(root)
    root.mainloop()
