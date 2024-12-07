import time
import numpy as np


# https://sebastianraschaka.com/pdf/lecture-notes/stat453ss21/L03_perceptron_slides.pdf
# https://sebastianraschka.com/blog/2020/numpy-intro.html
# https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html
# https://matplotlib.org/stable/tutorials/introductory/usage.html
def python_forloop_list_approach(x, w):
    z = 0.
    for i in range(len(x)):
        z+= x[i] * w[i]
    return z

a = [1., 2., 3.]
b = [4., 5., 6.]

start_time = time.time()  # Start the timer
result = python_forloop_list_approach(a, b)
end_time = time.time()    # End the timer

print(f"Result: {result}")
print(f"Execution time: {end_time - start_time:.6f} seconds")

large_a = list(range(1000))
large_b = list(range(1000))

# Measuring execution time for large inputs
start_time = time.time()  # Start the timer
result = python_forloop_list_approach(large_a, large_b)
end_time = time.time()    # End the timer

print(f"Result: {result}")
print(f"Execution time for large inputs: {end_time - start_time:.6f} seconds")

def numpy_dotproduct_approach(x, w):
    # np.dot(x, w)
    return x.dot(w)

a = np.array([1., 2., 3.])
b = np.array([4., 5., 6.])

# Measuring execution time with numpy
start_time = time.time()  # Start the timer
result = numpy_dotproduct_approach(a, b)
end_time = time.time()    # End the timer

print(f"Result: {result}")
print(f"Execution time for large inputs: {end_time - start_time:.6f} seconds")

# Convert large lists to NumPy arrays
large_a_np = np.array(large_a)
large_b_np = np.array(large_b)

# Measuring execution time with numpy but big array
start_time = time.time()  # Start the timer
result = numpy_dotproduct_approach(large_a_np, large_b_np)
end_time = time.time()    # End the timer

print(f"Result: {result}")
print(f"Execution time for large inputs: {end_time - start_time:.6f} seconds")