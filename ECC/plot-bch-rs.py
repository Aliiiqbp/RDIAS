import matplotlib.pyplot as plt

# Fixed x-axis list
x = [1, 2, 3, 4, 5, 6, 7, 8]

# Multiple lists to plot
y1 = [89/96, 82/96, 75/96, 68/96, 61/96, 54/96, 47/96, 40/96]
y2 = [80/95, 70/95, 60/95, 50/95, 40/95, 30/95, 20/95, 10/95]
# y3 = [5, 10, 15, 20, 25]

plt.figure(figsize=(12, 6))
# Plotting each list against the fixed x-axis
plt.plot(x, y1, label='BCH', marker='o', markersize=10, linestyle='-', color='b',linewidth=6)
plt.plot(x, y2, label='RS', marker='o', markersize=10, linestyle='-', color='r',linewidth=6)
# plt.plot(x, y3, label='y3: Multiples of 5')

# Adding labels and title
plt.xlabel('Bit Error Rate (%)', fontsize=32)
plt.xticks(fontsize=32)
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=32)
plt.ylabel('Efficiency', fontsize=32)
# plt.title('Multiple Lists Plotted Against a Fixed X-Axis')

# Adding a legend
plt.legend(fontsize=32)

# Display the plot
plt.grid(axis='y', color='gray', linestyle='--', linewidth=1, alpha=1)
plt.savefig("BCH-RS.pdf", format='pdf', bbox_inches='tight', pad_inches=0.01)
plt.show()
