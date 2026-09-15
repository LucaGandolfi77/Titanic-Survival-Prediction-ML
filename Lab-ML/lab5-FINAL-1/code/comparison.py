import matplotlib.pyplot as plt

from float_onemax import main as main_float
from bit_onemax import main as main_bits

pop_float, log_float, hof_float = main_float(0)
pop_bits, log_bits, hof_bits = main_bits(0)

gen_float = log_float.select("gen")
avg_float = log_float.select("avg")
best_float = log_float.select("max")

gen_bits = log_bits.select("gen")
avg_bits = log_bits.select("avg")
best_bits = log_bits.select("max")

plt.plot(gen_float, best_float, label="Float - Best")
plt.plot(gen_float, avg_float, label="Float - Average")

plt.plot(gen_bits, best_bits, label="Bits - Best")
plt.plot(gen_bits, avg_bits, label="Bits - Average")

plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.title("Convergence comparison")
plt.legend()
plt.grid()
plt.show()