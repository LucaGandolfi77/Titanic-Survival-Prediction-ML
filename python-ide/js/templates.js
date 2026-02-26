/* =====================================================
   TEMPLATES — Built-in code templates and examples
   ===================================================== */
window.IDE = window.IDE || {};

IDE.templates = (() => {
  'use strict';

  const TEMPLATES = [
    {
      name: 'Hello World',
      icon: 'fa-hand-wave',
      tip: 'Basic I/O with print() and input()',
      code: `# Hello World — basic I/O
print("Hello, World!")
name = input("What's your name? ")
print(f"Hello, {name}! Welcome to Python IDE.")
age = int(input("How old are you? "))
print(f"In 10 years you'll be {age + 10} years old.")
`
    },
    {
      name: 'Data Analysis with Pandas',
      icon: 'fa-table',
      tip: 'Create DataFrames, compute stats, filter data',
      code: `# Data Analysis with Pandas
import pandas as pd
import numpy as np

# Create a sample dataset
np.random.seed(42)
n = 200
data = {
    'Name': [f'Person_{i}' for i in range(n)],
    'Age': np.random.randint(18, 65, n),
    'Salary': np.random.normal(55000, 15000, n).astype(int),
    'Department': np.random.choice(['Engineering', 'Marketing', 'Sales', 'HR', 'Finance'], n),
    'Years_Experience': np.random.randint(0, 30, n)
}
df = pd.DataFrame(data)

# Display basic info
print("=== Dataset Overview ===")
print(f"Shape: {df.shape}")
print(f"\\nColumn types:\\n{df.dtypes}")

# Summary statistics
print("\\n=== Summary Statistics ===")
print(df.describe())

# Group by department
print("\\n=== Average Salary by Department ===")
dept_stats = df.groupby('Department').agg({
    'Salary': ['mean', 'median', 'std'],
    'Age': 'mean',
    'Years_Experience': 'mean'
}).round(2)
print(dept_stats)

# Filter high earners
high_earners = df[df['Salary'] > 70000].sort_values('Salary', ascending=False)
print(f"\\n=== High Earners (>{70000:,}) ===")
print(f"Count: {len(high_earners)}")
print(high_earners.head(10))
`
    },
    {
      name: 'Matplotlib Plotting',
      icon: 'fa-chart-line',
      tip: 'Generate and display plots with matplotlib',
      code: `# Matplotlib Plotting
import matplotlib.pyplot as plt
import numpy as np

# Create data
x = np.linspace(0, 4 * np.pi, 200)

# Figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle('Mathematical Functions', fontsize=14, fontweight='bold')

# Plot 1: Sin and Cos
axes[0, 0].plot(x, np.sin(x), label='sin(x)', color='#89b4fa', linewidth=2)
axes[0, 0].plot(x, np.cos(x), label='cos(x)', color='#f38ba8', linewidth=2)
axes[0, 0].set_title('Trigonometric')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Exponential
axes[0, 1].plot(x, np.exp(-x/5) * np.cos(x), color='#a6e3a1', linewidth=2)
axes[0, 1].set_title('Damped Oscillation')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Histogram
data = np.random.normal(0, 1, 1000)
axes[1, 0].hist(data, bins=30, color='#cba6f7', alpha=0.7, edgecolor='#1e1e2e')
axes[1, 0].set_title('Normal Distribution')
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Scatter
x_scatter = np.random.rand(100)
y_scatter = x_scatter + np.random.normal(0, 0.2, 100)
colors = np.random.rand(100)
axes[1, 1].scatter(x_scatter, y_scatter, c=colors, cmap='cool', alpha=0.7, s=50)
axes[1, 1].set_title('Scatter Plot')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
print("Plot rendered successfully!")
`
    },
    {
      name: 'Machine Learning (sklearn)',
      icon: 'fa-brain',
      tip: 'Full classification pipeline with scikit-learn',
      code: `# Machine Learning — Classification Pipeline
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
target_names = iris.target_names

print("=== Iris Dataset ===")
print(f"Features: {feature_names}")
print(f"Classes: {list(target_names)}")
print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"\\nTrain: {len(X_train)}, Test: {len(X_test)}")

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
accuracy = (y_pred == y_test).mean()
print(f"\\n=== Results ===")
print(f"Accuracy: {accuracy:.4f}")
print(f"\\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

# Cross-validation
cv_scores = cross_val_score(clf, X, y, cv=5)
print(f"Cross-validation scores: {cv_scores.round(4)}")
print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Feature importance
print(f"\\n=== Feature Importance ===")
for name, imp in sorted(zip(feature_names, clf.feature_importances_), key=lambda x: -x[1]):
    print(f"  {name:20s}: {imp:.4f} {'█' * int(imp * 40)}")
`
    },
    {
      name: 'Fibonacci (Recursion)',
      icon: 'fa-code-branch',
      tip: 'Recursive and iterative Fibonacci implementations',
      code: `# Fibonacci — Recursion vs Iteration
import time
from functools import lru_cache

# 1. Simple recursion (slow for large n)
def fib_recursive(n):
    if n <= 1:
        return n
    return fib_recursive(n - 1) + fib_recursive(n - 2)

# 2. Memoized recursion (fast)
@lru_cache(maxsize=None)
def fib_memo(n):
    if n <= 1:
        return n
    return fib_memo(n - 1) + fib_memo(n - 2)

# 3. Iterative (fastest, O(1) space)
def fib_iterative(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

# 4. Generator (lazy evaluation)
def fib_generator(limit):
    a, b = 0, 1
    for _ in range(limit):
        yield a
        a, b = b, a + b

# Compare performance
print("=== Fibonacci Comparison ===\\n")

# First 15 numbers
print("First 15 Fibonacci numbers:")
print(list(fib_generator(15)))

# Benchmark
for n in [10, 20, 30]:
    start = time.time()
    result = fib_recursive(n)
    t_rec = time.time() - start

    start = time.time()
    result = fib_memo(n)
    t_memo = time.time() - start

    start = time.time()
    result = fib_iterative(n)
    t_iter = time.time() - start

    print(f"\\nfib({n}) = {result}")
    print(f"  Recursive:  {t_rec*1000:.3f} ms")
    print(f"  Memoized:   {t_memo*1000:.3f} ms")
    print(f"  Iterative:  {t_iter*1000:.3f} ms")

# Large value with memoization
print(f"\\nfib(100) = {fib_memo(100)}")
print(f"fib(200) = {fib_memo(200)}")
`
    },
    {
      name: 'Web Scraping (Mocked)',
      icon: 'fa-globe',
      tip: 'Simulate web scraping with BeautifulSoup',
      code: `# Web Scraping — Simulated (runs in browser)
# Note: Real HTTP requests don't work in Pyodide,
# but we can demonstrate parsing HTML with BeautifulSoup

from bs4 import BeautifulSoup

# Simulated HTML content (as if fetched from a website)
html_content = """
<!DOCTYPE html>
<html>
<head><title>Sample Products Page</title></head>
<body>
  <h1>Tech Products Store</h1>
  <div class="products">
    <div class="product" data-id="1">
      <h2 class="name">Laptop Pro X1</h2>
      <span class="price">$1,299.00</span>
      <span class="rating">4.5/5</span>
      <p class="desc">High-performance laptop with 16GB RAM</p>
    </div>
    <div class="product" data-id="2">
      <h2 class="name">Wireless Mouse</h2>
      <span class="price">$49.99</span>
      <span class="rating">4.2/5</span>
      <p class="desc">Ergonomic wireless mouse with USB-C</p>
    </div>
    <div class="product" data-id="3">
      <h2 class="name">4K Monitor</h2>
      <span class="price">$599.00</span>
      <span class="rating">4.8/5</span>
      <p class="desc">27-inch 4K IPS display with HDR</p>
    </div>
    <div class="product" data-id="4">
      <h2 class="name">Mechanical Keyboard</h2>
      <span class="price">$129.99</span>
      <span class="rating">4.6/5</span>
      <p class="desc">Cherry MX switches, RGB backlit</p>
    </div>
  </div>
</body>
</html>
"""

soup = BeautifulSoup(html_content, 'html.parser')

# Extract page title
print(f"Page Title: {soup.title.string}")
print(f"Main Heading: {soup.h1.string}")
print()

# Extract all products
products = soup.find_all('div', class_='product')
print(f"Found {len(products)} products:\\n")

for p in products:
    name = p.find('h2', class_='name').text
    price = p.find('span', class_='price').text
    rating = p.find('span', class_='rating').text
    desc = p.find('p', class_='desc').text
    pid = p.get('data-id')
    print(f"[{pid}] {name}")
    print(f"    Price: {price} | Rating: {rating}")
    print(f"    {desc}")
    print()
`
    },
    {
      name: 'File I/O Demo',
      icon: 'fa-file-lines',
      tip: 'Read and write files using the virtual filesystem',
      code: `# File I/O — Virtual Filesystem Demo
import json
import csv
import os

# Write a text file
with open('/files/notes.txt', 'w') as f:
    f.write("Hello from Python IDE!\\n")
    f.write("This file is stored in the virtual filesystem.\\n")
    f.write("It persists in your browser's IndexedDB.\\n")

# Write a JSON file
data = {
    "users": [
        {"name": "Alice", "age": 30, "city": "New York"},
        {"name": "Bob", "age": 25, "city": "London"},
        {"name": "Charlie", "age": 35, "city": "Tokyo"}
    ],
    "total": 3,
    "version": "1.0"
}
with open('/files/users.json', 'w') as f:
    json.dump(data, f, indent=2)

# Write a CSV file
with open('/files/scores.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Name', 'Math', 'Science', 'English'])
    writer.writerow(['Alice', 95, 88, 92])
    writer.writerow(['Bob', 78, 95, 85])
    writer.writerow(['Charlie', 88, 92, 90])

# Read them back
print("=== Text File ===")
with open('/files/notes.txt', 'r') as f:
    print(f.read())

print("=== JSON File ===")
with open('/files/users.json', 'r') as f:
    loaded = json.load(f)
    for user in loaded['users']:
        print(f"  {user['name']}, age {user['age']}, from {user['city']}")

print("\\n=== CSV File ===")
with open('/files/scores.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        total = int(row['Math']) + int(row['Science']) + int(row['English'])
        print(f"  {row['Name']}: Math={row['Math']}, Science={row['Science']}, English={row['English']}, Total={total}")

# List files
print("\\n=== Files in /files/ ===")
for f in os.listdir('/files/'):
    size = os.path.getsize(f'/files/{f}')
    print(f"  {f} ({size} bytes)")
`
    },
    {
      name: 'Class & OOP Demo',
      icon: 'fa-diagram-project',
      tip: 'Classes, inheritance, properties, magic methods',
      code: `# Object-Oriented Programming Demo
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import List

# Abstract base class
class Shape(ABC):
    @abstractmethod
    def area(self) -> float: ...

    @abstractmethod
    def perimeter(self) -> float: ...

    def __str__(self):
        return f"{self.__class__.__name__}(area={self.area():.2f}, perimeter={self.perimeter():.2f})"

class Circle(Shape):
    def __init__(self, radius: float):
        self._radius = radius

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):
        if value <= 0:
            raise ValueError("Radius must be positive")
        self._radius = value

    def area(self) -> float:
        import math
        return math.pi * self._radius ** 2

    def perimeter(self) -> float:
        import math
        return 2 * math.pi * self._radius

class Rectangle(Shape):
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height

    def area(self) -> float:
        return self.width * self.height

    def perimeter(self) -> float:
        return 2 * (self.width + self.height)

# Dataclass
@dataclass
class Student:
    name: str
    grades: List[float] = field(default_factory=list)

    @property
    def average(self) -> float:
        return sum(self.grades) / len(self.grades) if self.grades else 0

    def add_grade(self, grade: float):
        self.grades.append(grade)
        return self

    def __str__(self):
        return f"{self.name}: avg={self.average:.1f}, grades={self.grades}"

# Demo
print("=== Shapes ===")
shapes = [Circle(5), Rectangle(4, 6), Circle(3), Rectangle(10, 2)]
for shape in shapes:
    print(f"  {shape}")

total_area = sum(s.area() for s in shapes)
print(f"\\nTotal area: {total_area:.2f}")

print("\\n=== Students ===")
students = [
    Student("Alice").add_grade(95).add_grade(88).add_grade(92),
    Student("Bob").add_grade(78).add_grade(85).add_grade(90),
    Student("Charlie").add_grade(92).add_grade(95).add_grade(88),
]
for s in students:
    print(f"  {s}")

best = max(students, key=lambda s: s.average)
print(f"\\nBest student: {best.name} (avg: {best.average:.1f})")
`
    },
    {
      name: 'Async/Await Demo',
      icon: 'fa-arrows-spin',
      tip: 'Asynchronous programming with asyncio',
      code: `# Async/Await Demo
import asyncio
import time

async def fetch_data(name, delay):
    """Simulate an async data fetch."""
    print(f"  [{name}] Starting fetch...")
    await asyncio.sleep(delay)
    result = f"Data from {name}"
    print(f"  [{name}] Done! ({delay}s)")
    return result

async def process_item(item, delay):
    """Simulate async processing."""
    await asyncio.sleep(delay)
    return f"Processed: {item.upper()}"

async def main():
    print("=== Concurrent Fetches ===")
    start = time.time()

    # Run multiple fetches concurrently
    results = await asyncio.gather(
        fetch_data("API-1", 0.5),
        fetch_data("API-2", 0.3),
        fetch_data("API-3", 0.7),
        fetch_data("API-4", 0.2),
    )

    elapsed = time.time() - start
    print(f"\\nAll fetches completed in {elapsed:.2f}s (not {0.5+0.3+0.7+0.2:.1f}s!)")
    print(f"Results: {results}")

    print("\\n=== Async Generator ===")
    async def countdown(n):
        for i in range(n, 0, -1):
            await asyncio.sleep(0.1)
            yield i

    async for num in countdown(5):
        print(f"  {num}...")
    print("  Liftoff!")

    print("\\n=== Task with Timeout ===")
    try:
        result = await asyncio.wait_for(fetch_data("Slow-API", 0.5), timeout=0.3)
    except asyncio.TimeoutError:
        print("  Task timed out! (as expected)")

    print("\\nAsync demo complete!")

asyncio.run(main())
`
    },
    {
      name: 'NumPy Array Operations',
      icon: 'fa-grip',
      tip: 'Array creation, operations, linear algebra with NumPy',
      code: `# NumPy Array Operations
import numpy as np

# Array creation
print("=== Array Creation ===")
a = np.array([1, 2, 3, 4, 5])
print(f"1D array: {a}")

b = np.arange(0, 20, 2)
print(f"arange(0,20,2): {b}")

c = np.linspace(0, 1, 5)
print(f"linspace(0,1,5): {c}")

d = np.random.randint(1, 100, (3, 4))
print(f"\\n3x4 random matrix:\\n{d}")

# Array operations
print("\\n=== Operations ===")
x = np.array([1, 2, 3, 4, 5])
print(f"x = {x}")
print(f"x * 2 = {x * 2}")
print(f"x ** 2 = {x ** 2}")
print(f"sum={x.sum()}, mean={x.mean():.1f}, std={x.std():.2f}")

# Matrix operations
print("\\n=== Linear Algebra ===")
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(f"A:\\n{A}")
print(f"B:\\n{B}")
print(f"A @ B (matmul):\\n{A @ B}")
print(f"det(A) = {np.linalg.det(A):.1f}")
print(f"inv(A):\\n{np.linalg.inv(A)}")

eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"\\nEigenvalues: {eigenvalues.round(4)}")

# Broadcasting
print("\\n=== Broadcasting ===")
grid = np.arange(9).reshape(3, 3)
col = np.array([10, 20, 30]).reshape(3, 1)
print(f"Grid:\\n{grid}")
print(f"Column: {col.flatten()}")
print(f"Grid + Column:\\n{grid + col}")

# Boolean indexing
print("\\n=== Fancy Indexing ===")
data = np.random.randn(10).round(3)
print(f"Data: {data}")
print(f"Positive values: {data[data > 0]}")
print(f"Count positive: {(data > 0).sum()}")
`
    },
    {
      name: 'Sorting Algorithms',
      icon: 'fa-sort',
      tip: 'Compare sorting algorithms by speed',
      code: `# Sorting Algorithms Comparison
import time
import random

def bubble_sort(arr):
    a = arr.copy()
    n = len(a)
    for i in range(n):
        for j in range(0, n - i - 1):
            if a[j] > a[j + 1]:
                a[j], a[j + 1] = a[j + 1], a[j]
    return a

def insertion_sort(arr):
    a = arr.copy()
    for i in range(1, len(a)):
        key = a[i]
        j = i - 1
        while j >= 0 and key < a[j]:
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = key
    return a

def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i]); i += 1
        else:
            result.append(right[j]); j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

# Benchmark
algorithms = {
    'Bubble Sort': bubble_sort,
    'Insertion Sort': insertion_sort,
    'Merge Sort': merge_sort,
    'Quick Sort': quick_sort,
    'Python sorted()': sorted
}

sizes = [100, 500, 1000, 2000]

print("=== Sorting Algorithm Benchmark ===")
print(f"{'Algorithm':<18}", end='')
for s in sizes:
    print(f"{'n='+str(s):>10}", end='')
print()
print("-" * 58)

for name, func in algorithms.items():
    print(f"{name:<18}", end='')
    for size in sizes:
        data = [random.randint(0, 10000) for _ in range(size)]
        start = time.time()
        result = func(data)
        elapsed = (time.time() - start) * 1000
        print(f"{elapsed:>8.1f}ms", end='')
        # Verify sort is correct
        assert result == sorted(data), f"{name} failed!"
    print()

print("\\nAll algorithms verified correct! ✓")
`
    },
    {
      name: 'Prime Number Sieve',
      icon: 'fa-hashtag',
      tip: 'Sieve of Eratosthenes and prime analysis',
      code: `# Prime Number Sieve (Eratosthenes)
import time
import math

def sieve_of_eratosthenes(limit):
    """Find all primes up to limit using the Sieve of Eratosthenes."""
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False

    for i in range(2, int(math.sqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False

    return [i for i in range(2, limit + 1) if is_prime[i]]

# Find primes
limits = [100, 1000, 10000, 100000, 1000000]
print("=== Sieve of Eratosthenes ===\\n")

for limit in limits:
    start = time.time()
    primes = sieve_of_eratosthenes(limit)
    elapsed = (time.time() - start) * 1000
    print(f"Primes up to {limit:>10,}: {len(primes):>8,} primes found ({elapsed:.1f}ms)")

print()

# Analyze primes up to 1000
primes = sieve_of_eratosthenes(1000)
print(f"First 20 primes: {primes[:20]}")
print(f"Last 10 primes under 1000: {primes[-10:]}")

# Twin primes (differ by 2)
twins = [(p, p+2) for p in primes if p+2 in set(primes)]
print(f"\\nTwin primes under 1000: {len(twins)}")
print(f"First 10 twin pairs: {twins[:10]}")

# Prime gaps
gaps = [primes[i+1] - primes[i] for i in range(len(primes)-1)]
print(f"\\nLargest prime gap under 1000: {max(gaps)}")
print(f"Average gap: {sum(gaps)/len(gaps):.2f}")

# Goldbach (even numbers as sum of two primes)
print("\\n=== Goldbach's Conjecture (even numbers 4-50) ===")
prime_set = set(primes)
for n in range(4, 52, 2):
    for p in primes:
        if p > n:
            break
        if (n - p) in prime_set:
            print(f"  {n} = {p} + {n - p}")
            break
`
    }
  ];

  function getAll() { return TEMPLATES; }

  function getByName(name) {
    return TEMPLATES.find(t => t.name === name);
  }

  function showPicker(onSelect) {
    let modal = document.getElementById('template-modal');
    if (modal) modal.remove();

    modal = document.createElement('div');
    modal.id = 'template-modal';
    modal.className = 'modal-overlay';
    modal.innerHTML = `
      <div class="modal-container modal-md">
        <div class="modal-header">
          <h2><i class="fa-solid fa-wand-magic-sparkles"></i> New from Template</h2>
          <button class="modal-close" id="template-close"><i class="fa-solid fa-xmark"></i></button>
        </div>
        <div class="modal-body">
          <div class="template-grid">
            ${TEMPLATES.map(t => `
              <div class="template-item" data-name="${t.name}" title="${t.tip}">
                <i class="fa-solid ${t.icon}"></i>
                <span>${t.name}</span>
                <small>${t.tip}</small>
              </div>
            `).join('')}
          </div>
        </div>
      </div>
    `;
    document.body.appendChild(modal);
    requestAnimationFrame(() => modal.classList.add('visible'));

    modal.querySelector('#template-close').addEventListener('click', () => modal.classList.remove('visible'));
    modal.addEventListener('click', e => { if (e.target === modal) modal.classList.remove('visible'); });

    modal.querySelectorAll('.template-item').forEach(item => {
      item.addEventListener('click', () => {
        const tmpl = getByName(item.dataset.name);
        if (tmpl && onSelect) onSelect(tmpl);
        modal.classList.remove('visible');
      });
    });
  }

  return { getAll, getByName, showPicker };
})();
