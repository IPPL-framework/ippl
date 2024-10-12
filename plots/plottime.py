import os
import re
import matplotlib.pyplot as plt

# Define the pattern for extracting timing values from the log file
time_patterns = {
    'Neighbor List Time': re.compile(r'Neighbor List Time: (\d+\.\d+)'),
    'PM Time': re.compile(r'PM Time: (\d+\.\d+)'),
    'PP Time': re.compile(r'PP Time: (\d+\.\d+)'),
    'Total Simulation time': re.compile(r'Total Simulation time: (\d+\.\d+)')
}

# Function to extract timing data from a log file
def extract_timing_data(log_file):
    timing_data = {}
    with open(log_file, 'r') as file:
        content = file.read()
        # Extract times for each pattern
        for key, pattern in time_patterns.items():
            match = pattern.search(content)
            if match:
                timing_data[key] = float(match.group(1))
            else:
                timing_data[key] = None  # Handle cases where the time isn't found
    return timing_data

# Function to read all log files and store timing data
def read_log_files(log_directory):
    timing_results = {}
    # Loop over files in directory
    for filename in os.listdir(log_directory):
        if filename.startswith("TimeHeating") and filename.endswith(".log"):
            # Extract the variable X from the filename
            run_id = filename[len("TimeHeating"):-len(".log")]
            # Extract the timing data from the file
            timing_data = extract_timing_data(os.path.join(log_directory, filename))
            timing_results[run_id] = timing_data
    return timing_results

# Function to plot timing data
def plot_timing_results(timing_results):
    # Prepare data for plotting
    runs = sorted(timing_results.keys())  # Sort by run_id (X)
    neighbor_list_times = [timing_results[run]['Neighbor List Time'] for run in runs]
    pm_times = [timing_results[run]['PM Time'] for run in runs]
    pp_times = [timing_results[run]['PP Time'] for run in runs]
    total_simulation_times = [timing_results[run]['Total Simulation time'] for run in runs]

    # Create subplots for each type of time
    plt.figure(figsize=(10, 6))
    
    # Plot Neighbor List Time
    plt.plot(runs, neighbor_list_times, marker='o', label='Neighbor List Time', color='r')
    
    # Plot PM Time
    plt.plot(runs, pm_times, marker='o', label='PM Time', color='g')
    
    # Plot PP Time
    plt.plot(runs, pp_times, marker='o', label='PP Time', color='b')
    
    # Plot Total Simulation Time
    plt.plot(runs, total_simulation_times, marker='o', label='Total Simulation Time', color='k')

    # Add labels and legend
    plt.xlabel('Run ID (X)')
    plt.ylabel('Time (seconds)')
    plt.title('Timing Results for Multiple Simulation Runs')
    plt.legend()
    plt.grid(True)
    
    # Show plot
    plt.tight_layout()
    plt.savefig('timings.pdf')

# Main script execution
if __name__ == "__main__":
    # Path to the directory containing log files
    log_directory = './out'  # Modify this path to where your log files are stored
    
    # Read log files and extract timing data
    timing_results = read_log_files(log_directory)
    
    # Plot the timing results
    plot_timing_results(timing_results)
