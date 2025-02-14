import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.optimize import root_scalar
from sympy import symbols, solve
from tkinter import simpledialog
from scipy import interpolate as interp


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
            ("Power Method for Eigenvalues", self.task4),
            ("Exponential Curve Fitting", self.task5_curve_fitting),
            ("Cubic Spline Interpolation", self.task6),
            ("Modified Euler’s Method", self.task7),
            ("Weddle’s Rule Integration", self.task8)
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

        result_label = ttk.Label(task_window, text="", font=("Arial", 12))
        result_label.pack()

        canvas_frame = ttk.Frame(task_window)
        canvas_frame.pack()

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

                # Проверка на корень
                if f(a) * f(b) > 0:
                    messagebox.showerror("No Root",
                                         "Function values at a and b have the same sign. No guarantee of a root.")
                    return

                root_fp, iter_fp = self.false_position_method(f, a, b)
                root_nr, iter_nr = self.newton_method(f, df, x0)

                # Формирование текста результата
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

                # Обновление текста в result_label вместо создания нового Label
                result_label.config(text=result_text)

                # Очистка предыдущего графика, если есть
                for widget in canvas_frame.winfo_children():
                    widget.destroy()

                # Построение нового графика
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

                canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
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

        # Label for result
        result_label = ttk.Label(task_window, text="", font=("Arial", 12))
        result_label.pack()

        # Frame for graph
        canvas_frame = ttk.Frame(task_window)
        canvas_frame.pack()

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

                if iterations < 200:
                    result_text = f"Solution: {solution}\nIterations: {iterations}"
                else:
                    result_text = "Method did not converge within 100 iterations."

                result_label.config(text=result_text)

                for widget in canvas_frame.winfo_children():
                    widget.destroy()

                fig, ax = plt.subplots(figsize=(5, 3))
                ax.plot(range(len(errors)), errors, marker='o', linestyle='-', color='b', label="Error per iteration")
                ax.set_yscale("log")  # Log scale for better visualization
                ax.set_xlabel("Iteration")
                ax.set_ylabel("Error (||x_new - x_old||)")
                ax.legend()
                ax.grid()

                canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
                canvas.draw()
                canvas.get_tk_widget().pack()

            except ValueError:
                messagebox.showerror("Input Error", "Please enter a valid numerical value for ω.")
            except Exception as e:
                messagebox.showerror("Error", f"Unexpected error: {str(e)}")

        ttk.Button(task_window, text="Compute", command=compute).pack(pady=5)

    def relaxation_method(self, A, b, omega=0.9, tol=1e-6, max_iter=100):
        n = len(b)
        x = np.zeros(n)
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

    def task4(self):
        task_window = tk.Toplevel(self.root)
        task_window.title("Power Method for Eigenvalues")
        self.center_window(task_window, 800, 600)

        ttk.Label(task_window, text="Power Method for Eigenvalues", font=("Arial", 12)).pack()

        ttk.Label(task_window, text="Enter matrix size (n x n):", font=("Arial", 10)).pack()
        size_entry = tk.Entry(task_window, font=("Arial", 12), width=5, justify="center")
        size_entry.insert(0, "3")  # Default
        size_entry.pack()

        matrix_frame = ttk.Frame(task_window)
        matrix_frame.pack()

        def create_matrix_input(n):
            for widget in matrix_frame.winfo_children():
                widget.destroy()

            self.matrix_entries = []
            for i in range(n):
                row_entries = []
                for j in range(n):
                    entry = tk.Entry(matrix_frame, width=5, font=("Arial", 12), justify="center")
                    entry.grid(row=i, column=j, padx=2, pady=2)
                    entry.insert(0, str(1 if i == j else 0))
                    row_entries.append(entry)
                self.matrix_entries.append(row_entries)

        def get_matrix():
            try:
                return np.array([[float(entry.get()) for entry in row] for row in self.matrix_entries])
            except ValueError:
                messagebox.showerror("Input Error", "Invalid matrix values.")
                return None

        # Button for input
        def update_matrix():
            try:
                n = int(size_entry.get())
                create_matrix_input(n)
            except ValueError:
                messagebox.showerror("Input Error", "Invalid size.")

        ttk.Button(task_window, text="Set Matrix Size", command=update_matrix).pack(pady=5)
        create_matrix_input(3)

        # Initia; vector
        ttk.Label(task_window, text="Enter initial vector x0 (comma-separated):", font=("Arial", 10)).pack()
        x0_entry = tk.Entry(task_window, font=("Arial", 12))
        x0_entry.insert(0, "1,1,1")  # default
        x0_entry.pack()

        result_label = ttk.Label(task_window, text="", font=("Arial", 12))
        result_label.pack()

        canvas_frame = ttk.Frame(task_window)
        canvas_frame.pack()

        def compute():
            A = get_matrix()
            if A is None:
                return

            try:
                x0 = np.array(list(map(float, x0_entry.get().split(','))))
                if len(x0) != A.shape[0]:
                    raise ValueError("Vector size must match matrix size.")
            except ValueError:
                messagebox.showerror("Input Error", "Invalid initial vector.")
                return

            eigenvalue, eigenvector, iterations = self.power_method(A, x0)
            result_text = f"Eigenvalue: {eigenvalue:.6f}\nEigenvector: {eigenvector}\nIterations: {iterations}"
            result_label.config(text=result_text)

            for widget in canvas_frame.winfo_children():
                widget.destroy()

            fig, ax = plt.subplots(figsize=(5, 3))
            ax.plot(range(len(eigenvector)), eigenvector, marker='o', linestyle='-', color='b', label="Eigenvector")
            ax.set_xlabel("Component Index")
            ax.set_ylabel("Eigenvector Components")
            ax.legend()
            ax.grid()

            canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
            canvas.draw()
            canvas.get_tk_widget().pack()

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

    def task6(self):
        task_window = tk.Toplevel(self.root)
        task_window.title("Cubic Spline Interpolation")
        self.center_window(task_window)

        # Исходные точки для интерполяции
        x_data = np.array([0.5, 1.5, 2.5, 3.5])
        y_data = np.array([0.25, 0.75, 2.25, 6.25])

        # Интерполяция с использованием кубического сплайна
        spline = interp.CubicSpline(x_data, y_data)

        # Запрашиваем у пользователя точки для оценки
        ttk.Label(task_window, text="Enter two x values for estimation (comma-separated):", font=("Arial", 10)).pack()
        x_input = tk.Entry(task_window, font=("Arial", 12))
        x_input.pack()

        def compute():
            try:
                # Получаем два значения x от пользователя
                x_values = np.array([float(x) for x in x_input.get().split(',')])
                
                if len(x_values) != 2:
                    messagebox.showerror("Input Error", "Please enter exactly two x values.")
                    return
                
                # Вычисляем значения y для введённых x
                y_values = spline(x_values)
                result_text = "\n".join([f"x = {x:.2f}, y = {y:.4f}" for x, y in zip(x_values, y_values)])

                ttk.Label(task_window, text=result_text, font=("Arial", 12)).pack()

                # График интерполяции
                x_range = np.linspace(0.5, 3.5, 100)
                y_range = spline(x_range)

                plt.plot(x_range, y_range, label='Cubic Spline')
                plt.scatter(x_data, y_data, color='red', label='Data Points')
                plt.xlabel("x")
                plt.ylabel("y")
                plt.legend()
                plt.grid()

                plt.show()

            except ValueError:
                messagebox.showerror("Input Error", "Please enter valid x values.")
            except Exception as e:
                messagebox.showerror("Error", f"Unexpected error: {str(e)}")

        ttk.Button(task_window, text="Compute", command=compute).pack(pady=5)



    def task7(self):
        task_window = tk.Toplevel(self.root)
        task_window.title("Modified Euler's Method")
        self.center_window(task_window)

        # Уравнение dy/dx = sin(x) - y
        def f(x, y):
            return np.sin(x) - y

        # Модифицированный метод Эйлера
        def modified_euler(f, y0, x0, x_end, h):
            x = x0
            y = y0
            x_values = [x]  # Список для значений x
            y_values = [y]  # Список для значений y
            while x < x_end:
                y_predict = y + h * f(x, y)
                y_corrected = y + h * (f(x + h, y_predict) + f(x, y)) / 2
                x += h
                y = y_corrected
                x_values.append(x)
                y_values.append(y)
            return x_values, y_values

        # Ввод параметров
        ttk.Label(task_window, text="Enter the step size (h):", font=("Arial", 10)).pack()
        h_entry = tk.Entry(task_window, font=("Arial", 12))
        h_entry.pack()

        def compute():
            try:
                h = float(h_entry.get())  # Получаем значение шага
                if h <= 0:
                    messagebox.showerror("Input Error", "Step size must be positive.")  # Ошибка если шаг отрицателен
                    return

                # Вычисление x и y для графика
                x_values, y_values = modified_euler(f, y0=1, x0=0, x_end=0.4, h=h)
                result_text = f"y(0.4) = {y_values[-1]:.6f}"  # Форматируем результат

                ttk.Label(task_window, text=result_text, font=("Arial", 12)).pack()

                # График
                plt.plot(x_values, y_values, label="y(x) - Modified Euler's")
                plt.xlabel("x")
                plt.ylabel("y")
                plt.title("Modified Euler's Method")
                plt.grid(True)
                plt.legend()
                plt.show()

            except ValueError:
                messagebox.showerror("Input Error", "Please enter a valid step size.")  # Ошибка при неправильном вводе
            except Exception as e:
                messagebox.showerror("Error", f"Unexpected error: {str(e)}")

        ttk.Button(task_window, text="Compute", command=compute).pack(pady=5)

    def task8(self):
        task_window = tk.Toplevel(self.root)
        task_window.title("Weddle's Rule")
        self.center_window(task_window)

        ttk.Label(task_window, text="Enter the number of subintervals (n):", font=("Arial", 10)).pack()
        n_entry = tk.Entry(task_window, font=("Arial", 12))
        n_entry.pack()

        def weddle_rule(f, a, b, n):
            if n % 6 != 0:
                n += 6 - (n % 6)  # Приводим n к ближайшему, кратному 6
            h = (b - a) / n
            integral = 0
            for i in range(0, n, 6):
                x0 = a + i * h
                x1, x2, x3, x4, x5, x6 = [x0 + j * h for j in range(1, 7)]
                
                integral += (3 * h / 10) * (
                    f(x0) + 5 * f(x1) + f(x2) + 6 * f(x3) + f(x4) + 5 * f(x5) + f(x6)
                )
            return integral

        def f(x):
            return 1 / (1 + x**2)

        def compute():
            try:
                n = int(n_entry.get())
                if n < 6:
                    messagebox.showerror("Input Error", "Number of subintervals must be at least 6.")
                    return

                result = weddle_rule(f, a=0, b=6, n=n)
                result_text = f"Integral = {result:.6f}"
                ttk.Label(task_window, text=result_text, font=("Arial", 12)).pack()

                # График функции
                x_vals = np.linspace(0, 6, 100)
                y_vals = f(x_vals)

                plt.figure(figsize=(8, 5))
                plt.plot(x_vals, y_vals, label=r'$f(x) = \frac{1}{1+x^2}$', color='blue')
                plt.fill_between(x_vals, y_vals, alpha=0.3, color='cyan', label="Area under curve")

                # Отмечаем узлы Weddle
                x_nodes = np.linspace(0, 6, n + 1)
                y_nodes = f(x_nodes)
                plt.scatter(x_nodes, y_nodes, color='red', label='Weddle Nodes')

                plt.xlabel("x")
                plt.ylabel("f(x)")
                plt.title("Weddle’s Rule Approximation of Integral")
                plt.legend()
                plt.grid(True)

                plt.show()

            except ValueError:
                messagebox.showerror("Input Error", "Please enter a valid number of subintervals.")
            except Exception as e:
                messagebox.showerror("Error", f"Unexpected error: {str(e)}")

        ttk.Button(task_window, text="Compute", command=compute).pack(pady=5)


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
