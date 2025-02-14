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
            ("Comparison of Root-Finding Methods", self.task2),
            ("Relaxation Method", self.task3),
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

    def task2(self):
        task_window = tk.Toplevel(self.root)
        task_window.title("Comparison of Root-Finding Methods")
        self.center_window(task_window)

        ttk.Label(task_window, text="Finding root of f(x) = ln(x) - x/10", font=("Arial", 12)).pack()

        ttk.Label(task_window, text="Enter interval [a, b]:", font=("Arial", 10)).pack()

        a_entry = tk.Entry(task_window, font=("Arial", 12))
        a_entry.pack()
        b_entry = tk.Entry(task_window, font=("Arial", 12))
        b_entry.pack()

        def compute():
            try:
                a = float(a_entry.get())
                b = float(b_entry.get())

                if a <= 0 or b <= 0 or a >= b:
                    messagebox.showerror("Input Error", "Enter a valid interval: 0 < a < b")
                    return

                f = lambda x: np.log(x) - x / 10
                df = lambda x: 1 / x - 0.1
                x0 = (a + b) / 2

                # Checl for root
                if f(a) * f(b) > 0:
                    messagebox.showerror("No Root",
                                         "Function values at a and b have the same sign. No guarantee of a root.")
                    return

                root_fp, iter_fp = self.false_position_method(f, a, b)

                # Validate result of False Position
                if root_fp is not None:
                    print(f"False Position Method found root: {root_fp:.6f} in {iter_fp} iterations")

                # Checck for errors with df(x)
                if abs(df(x0)) < 1e-6:
                    print("Warning: Derivative is too small, Newton's method may fail.")

                # Calling Newton method
                root_nr, iter_nr = self.newton_method(f, df, x0)

                # Result of Newton-Raphson
                if root_nr is None:
                    print("Newton-Raphson Method failed.")
                else:
                    print(f"Newton-Raphson Method found root: {root_nr:.6f} in {iter_nr} iterations")

                if root_fp is None and root_nr is None:
                    result_text = "Failed to find root using both methods."
                elif root_fp is None:
                    result_text = (f"Newton-Raphson Method: Root = {root_nr:.6f}, Iterations = {iter_nr}\n"
                                   f"False Position Method failed.")
                elif root_nr is None:
                    result_text = (f"False Position Method: Root = {root_fp:.6f}, Iterations = {iter_fp}\n"
                                   f"Newton-Raphson Method failed.")
                else:
                    rel_error = abs(root_fp - root_nr) / abs(root_nr) * 100
                    result_text = (f"False Position Method: Root = {root_fp:.6f}, Iterations = {iter_fp}\n"
                                   f"Newton-Raphson Method: Root = {root_nr:.6f}, Iterations = {iter_nr}\n"
                                   f"Relative Error = {rel_error:.6f}%")

                ttk.Label(task_window, text=result_text, font=("Arial", 12)).pack()

                # Plot the graph
                x_vals = np.linspace(a, b, 100)
                y_vals = f(x_vals)

                fig, ax = plt.subplots(figsize=(5, 3))
                ax.plot(x_vals, y_vals, label='f(x) = ln(x) - x/10', color='b')
                ax.axhline(0, color='black', linewidth=0.5)
                if root_fp:
                    ax.plot(root_fp, f(root_fp), 'ro', label='False Position Root')
                if root_nr:
                    ax.plot(root_nr, f(root_nr), 'go', label='Newton-Raphson Root')
                ax.legend()
                ax.grid()

                # Draw a graph on window
                canvas = FigureCanvasTkAgg(fig, master=task_window)
                canvas.draw()
                canvas.get_tk_widget().pack()

            except ValueError:
                messagebox.showerror("Input Error", "Please enter valid numerical values.")
            except Exception as e:
                messagebox.showerror("Error", f"Unexpected error: {str(e)}")

        ttk.Button(task_window, text="Compute", command=compute).pack(pady=5)

    def false_position_method(self, f, a, b, tol=1e-6, max_iter=100):

        fa, fb = f(a), f(b)

        # Ensure the initial interval contains a root
        if fa * fb > 0:
            print("False Position Method failed: No sign change in the interval.")
            return None, 0

        for i in range(max_iter):
            # Compute the new approximation
            c = (a * fb - b * fa) / (fb - fa)
            fc = f(c)

            # Check for convergence
            if abs(fc) < tol:
                return c, i + 1

            # Update interval based on sign of f(c)
            if fa * fc < 0:
                b, fb = c, fc  # Root lies in [a, c], update b
            else:
                a, fa = c, fc  # Root lies in [c, b], update a

        print("False Position Method did not converge within the maximum number of iterations.")
        return None, max_iter

    def newton_method(self, f, df, x0, tol=1e-6, max_iter=100):
        for i in range(max_iter):
            fx = f(x0)
            dfx = df(x0)

            # Check if the derivative is too small
            if abs(dfx) < 1e-8:
                print(f"Newton's method failed: derivative too small at iteration {i}, x = {x0}")
                return None, i

            # Check if the method has converged
            if abs(fx) < tol:
                return x0, i + 1

            # Compute the next approximation
            x_next = x0 - fx / dfx

            # Limit the step size
            if abs(x_next - x0) > 1:
                x_next = x0 - np.sign(fx / dfx) * 1  # Restricting step size

            # Ensure x remains in the valid domain
            if x_next <= 0:
                print(f"Newton's method failed: x went out of domain (x = {x_next}) at iteration {i}")
                return None, i

            x0 = x_next  # Update x

        print("Newton's method did not converge within the maximum number of iterations.")
        return None, max_iter

    def task3(self):
        task_window = tk.Toplevel(self.root)
        task_window.title("Relaxation Method (SOR)")
        self.center_window(task_window)

        ttk.Label(task_window, text="Solving system using Relaxation Method (SOR)", font=("Arial", 12)).pack()

        ttk.Label(task_window, text="Enter relaxation parameter (ω):", font=("Arial", 10)).pack()
        omega_entry = tk.Entry(task_window, font=("Arial", 12))
        omega_entry.insert(0, "0.9")  # Default value
        omega_entry.pack()

        def compute():
            try:
                # Read input
                omega = float(omega_entry.get())

                if not (0 < omega < 2):
                    messagebox.showerror("Input Error", "Enter a valid ω in range (0, 2).")
                    return

                # Define system
                A = np.array([[1, 1, 1],
                              [2, -3, 4],
                              [3, 4, 5]], dtype=float)
                b = np.array([9, 13, 40], dtype=float)

                # Solve with relaxation method
                solution, iterations, log, errors = self.relaxation_method(A, b, omega)

                # Display results
                if iterations < 200:
                    result_text = f"Solution: {solution}\nIterations: {iterations}"
                else:
                    result_text = "Method did not converge within 100 iterations."

                ttk.Label(task_window, text=result_text, font=("Arial", 12)).pack()

                # Plot convergence graph
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.plot(range(len(errors)), errors, marker='o', linestyle='-', color='b', label="Error per iteration")
                ax.set_yscale("log")  # Log scale for better visualization
                ax.set_xlabel("Iteration")
                ax.set_ylabel("Error (||x_new - x_old||)")
                ax.legend()
                ax.grid()

                canvas = FigureCanvasTkAgg(fig, master=task_window)
                canvas.draw()
                canvas.get_tk_widget().pack()

            except ValueError:
                messagebox.showerror("Input Error", "Please enter a valid numerical value for ω.")
            except Exception as e:
                messagebox.showerror("Error", f"Unexpected error: {str(e)}")

        ttk.Button(task_window, text="Compute", command=compute).pack(pady=5)

    def relaxation_method(self, A, b, omega=0.9, tol=1e-6, max_iter=100):
        """ Метод релаксации (SOR) для решения системы уравнений. """
        n = len(b)
        x = np.zeros(n)  # Начальное приближение
        log = ""
        errors = []

        for iter_count in range(max_iter):
            x_new = np.copy(x)

            for i in range(n):
                sigma = sum(A[i, j] * x_new[j] for j in range(n) if j != i)
                x_new[i] = (1 - omega) * x[i] + (omega / A[i, i]) * (b[i] - sigma)

            error = np.linalg.norm(x_new - x, ord=np.inf)
            errors.append(error)
            log += f"Iter {iter_count + 1}: {x_new}, Error: {error:.2e}\n"

            if error < tol:
                return x_new, iter_count + 1, log, errors

            x = x_new

        return x, max_iter, log, errors  # If no convergence

# ...existing code...

    def task4_power_method(self):
        task_window = tk.Toplevel(self.root)
        task_window.title("Power Method for Eigenvalues")
        self.center_window(task_window)

        ttk.Label(task_window, text="Power Method for Eigenvalues", font=("Arial", 12)).pack()

        ttk.Label(task_window, text="Enter matrix A (comma-separated rows):", font=("Arial", 10)).pack()
        A_entry = tk.Entry(task_window, font=("Arial", 12))
        A_entry.insert(0, "8,4,2;4,8,4;2,4,8")  # Default example
        A_entry.pack()

        ttk.Label(task_window, text="Enter initial vector x0 (comma-separated):", font=("Arial", 10)).pack()
        x0_entry = tk.Entry(task_window, font=("Arial", 12))
        x0_entry.insert(0, "1,1,1")  # Default initial vector
        x0_entry.pack()

        result_frame = tk.Frame(task_window)
        result_frame.pack()

        def compute():
            try:
                A = np.array([list(map(float, row.split(','))) for row in A_entry.get().split(';')])
                x0 = np.array(list(map(float, x0_entry.get().split(','))))

                eigenvalue, eigenvector, iterations = self.power_method(A, x0)

                result_text = f"Eigenvalue: {eigenvalue}\nEigenvector: {eigenvector}\nIterations: {iterations}"
                ttk.Label(result_frame, text=result_text, font=("Arial", 12)).pack()

                # Plot the convergence of the eigenvector
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.plot(range(len(eigenvector)), eigenvector, marker='o', linestyle='-', color='b', label="Eigenvector")
                ax.set_xlabel("Iteration")
                ax.set_ylabel("Eigenvector Components")
                ax.legend()
                ax.grid()

                canvas = FigureCanvasTkAgg(fig, master=result_frame)
                canvas.draw()
                canvas.get_tk_widget().pack()

            except ValueError:
                messagebox.showerror("Input Error", "Please enter valid numerical values.")
            except Exception as e:
                messagebox.showerror("Error", f"Unexpected error: {str(e)}")

        ttk.Button(task_window, text="Compute", command=compute).pack(pady=5)

    def power_method(self, A, x0, tol=1e-6, max_iter=100):
        x = x0
        for i in range(max_iter):
            x_new = np.dot(A, x)
            x_new_norm = np.linalg.norm(x_new)
            x_new = x_new / x_new_norm

            if np.linalg.norm(x_new - x) < tol:
                eigenvalue = np.dot(x_new, np.dot(A, x_new))
                return eigenvalue, x_new, i + 1

            x = x_new

        eigenvalue = np.dot(x, np.dot(A, x))
        return eigenvalue, x, max_iter

    def task5_curve_fitting(self):
        task_window = tk.Toplevel(self.root)
        task_window.title("Exponential Curve Fitting")
        self.center_window(task_window)

        ttk.Label(task_window, text="Exponential Curve Fitting", font=("Arial", 12)).pack()

        ttk.Label(task_window, text="Enter x values (comma-separated):", font=("Arial", 10)).pack()
        x_entry = tk.Entry(task_window, font=("Arial", 12))
        x_entry.insert(0, "0.5,1.5,2.5,3.5")  # Default example
        x_entry.pack()

        ttk.Label(task_window, text="Enter y values (comma-separated):", font=("Arial", 10)).pack()
        y_entry = tk.Entry(task_window, font=("Arial", 12))
        y_entry.insert(0, "2,6,18,54")  # Default example
        y_entry.pack()

        result_frame = tk.Frame(task_window)
        result_frame.pack()

        def compute():
            try:
                x = np.array(list(map(float, x_entry.get().split(','))))
                y = np.array(list(map(float, y_entry.get().split(','))))

                popt, pcov = self.exponential_curve_fitting(x, y)
                a, b = popt

                result_text = f"Fitted Parameters: a = {a}, b = {b}"
                ttk.Label(result_frame, text=result_text, font=("Arial", 12)).pack()

                # Plot the data and the fitted curve
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.scatter(x, y, label='Data', color='b')
                ax.plot(x, a * np.exp(b * x), label='Fitted Curve', color='r')
                ax.legend()
                ax.grid()

                canvas = FigureCanvasTkAgg(fig, master=result_frame)
                canvas.draw()
                canvas.get_tk_widget().pack()

            except ValueError:
                messagebox.showerror("Input Error", "Please enter valid numerical values.")
            except Exception as e:
                messagebox.showerror("Error", f"Unexpected error: {str(e)}")

        ttk.Button(task_window, text="Compute", command=compute).pack(pady=5)

    def exponential_curve_fitting(self, x, y):
        from scipy.optimize import curve_fit

        def model(x, a, b):
            return a * np.exp(b * x)

        popt, pcov = curve_fit(model, x, y)
        return popt, pcov

# ...existing code...

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
