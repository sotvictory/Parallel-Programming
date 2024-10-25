import matplotlib.pyplot as plt
import re

def read_log_file(filename):
    results = {
        'P': [],
        'Duration Serial': [],
        'Duration Parallel': [],
        'Speedup': [],
        'Computational Complexity W': [],
        'Efficiency': [],
        'Cost of Computations C': [],
        'Total Overhead T0': []
    }

    with open(filename, 'r') as file:
        content = file.read().strip().split('\n\n')

        for block in content:
            lines = block.strip().split('\n')
            if len(lines) < 8:
                continue

            p_match = re.search(r'P: (\d+)', lines[0])
            duration_serial_match = re.search(r'Duration Serial: ([\d.]+)', lines[1])
            duration_parallel_match = re.search(r'Duration Parallel: ([\d.]+)', lines[2])
            speedup_match = re.search(r'Speedup S: ([\d.]+)', lines[3])
            computational_complexity_match = re.search(r'Computational complexity W: ([\d.]+)', lines[4])
            efficiency_match = re.search(r'Efficiency E: ([\d.]+)', lines[5])
            cost_of_computations_match = re.search(r'Cost of computations C: ([\d.]+)', lines[6])
            total_overhead_match = re.search(r'Total overhead T0: ([\d.]+)', lines[7])

            if (p_match and duration_serial_match and duration_parallel_match and 
                speedup_match and efficiency_match and cost_of_computations_match and 
                total_overhead_match and computational_complexity_match):
                
                results['P'].append(int(p_match.group(1)))
                results['Duration Serial'].append(float(duration_serial_match.group(1)))
                results['Duration Parallel'].append(float(duration_parallel_match.group(1)))
                results['Speedup'].append(float(speedup_match.group(1)))
                results['Computational Complexity W'].append(float(computational_complexity_match.group(1)))
                results['Efficiency'].append(float(efficiency_match.group(1)))
                results['Cost of Computations C'].append(float(cost_of_computations_match.group(1)))
                results['Total Overhead T0'].append(float(total_overhead_match.group(1)))

    return results

def plot_performance_metrics(data):
    plt.style.use('default')
    fig, axs = plt.subplots(3, 2, figsize=(12, 12))

    max_tests = 5
    P_data = data['P'][:max_tests]
    speedup_data = data['Speedup'][:max_tests]
    efficiency_data = data['Efficiency'][:max_tests]
    duration_serial_data = data['Duration Serial'][:max_tests]
    duration_parallel_data = data['Duration Parallel'][:max_tests]
    cost_data = data['Cost of Computations C'][:max_tests]
    overhead_data = data['Total Overhead T0'][:max_tests]

    # 1: Speedup
    axs[0, 0].plot(P_data, speedup_data, marker='o')
    axs[0, 0].set_title('Speedup')
    axs[0, 0].set_xlabel('Number of Processes (P)')
    axs[0, 0].set_ylabel('Speedup (S)')
    axs[0, 0].grid(True, linestyle='--', linewidth=0.7) 

    # 2: Efficiency
    axs[0, 1].plot(P_data, efficiency_data, marker='o', color='orange')
    axs[0, 1].set_title('Efficiency')
    axs[0, 1].set_xlabel('Number of Processes (P)')
    axs[0, 1].set_ylabel('Efficiency (E)')
    axs[0, 1].grid(True, linestyle='--', linewidth=0.7)

    # 3: Duration Serial vs Duration Parallel
    axs[1, 0].plot(P_data, duration_serial_data, marker='o', label='Serial', color='green')
    axs[1, 0].plot(P_data, duration_parallel_data, marker='o', label='Parallel', color='red')
    axs[1, 0].set_title('Duration: Serial vs Parallel')
    axs[1, 0].set_xlabel('Number of Processes (P)')
    axs[1, 0].set_ylabel('Duration (seconds)')
    axs[1, 0].legend()
    axs[1, 0].grid(True, linestyle='--', linewidth=0.7)

    # 4: Cost of Computations
    axs[1, 1].plot(P_data, cost_data, marker='o', color='purple')
    axs[1, 1].set_title('Cost of Computations')
    axs[1, 1].set_xlabel('Number of Processes (P)')
    axs[1, 1].set_ylabel('Cost of Computations (C)')
    axs[1, 1].grid(True, linestyle='--', linewidth=0.7) 

    # 5: Total Overhead
    axs[2, 0].plot(P_data, overhead_data, marker='o', color='brown')
    axs[2, 0].set_title('Total Overhead')
    axs[2, 0].set_xlabel('Number of Processes (P)')
    axs[2, 0].set_ylabel('Total Overhead (T₀)')
    axs[2, 0].grid(True, linestyle='--', linewidth=0.7)

    fig.delaxes(axs[2, 1])

    plt.tight_layout()
    plt.savefig('performance_metrics.png')
    plt.close(fig)

def plot_scalability(data):
    plt.style.use('default')

    strong_P = data['P'][:5]
    strong_speedup = data['Speedup'][:5]
    
    weak_P = data['P'][-5:]
    weak_speedup = data['Speedup'][-5:]

    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    # Strong Scalability
    axs[0].plot(strong_P, strong_speedup, marker='o', color='blue')
    axs[0].set_title('Strong Scalability')
    axs[0].set_xlabel('Number of Processes (P)')
    axs[0].set_ylabel('Speedup (S)')
    axs[0].grid(True, linestyle='--', linewidth=0.7)

    # Weak Scalability
    axs[1].plot(weak_P, weak_speedup, marker='o', color='orange')
    axs[1].set_title('Weak Scalability')
    axs[1].set_xlabel('Number of Processes (P)')
    axs[1].set_ylabel('Speedup (S)')
    axs[1].grid(True, linestyle='--', linewidth=0.7)

    plt.tight_layout()
    plt.savefig('scalability_metrics.png')
    plt.close(fig)

data = read_log_file('log.txt')

if data:
    plot_scalability(data)