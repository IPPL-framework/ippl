import re
import matplotlib.pyplot as plt

# Function to parse the data from the simulation output file
def parse_simulation_data(file_path):
    temperature_data = []
    rms_beam_size_data = []
    rms_emittance_data = []
    l2_temperature_data = []

    with open(file_path, 'r') as file:
        content = file.read()

    timestep_data = re.split(r'LeapFrog Step \d+ Finished\.', content)
    for timestep in timestep_data:
        # Parse Temperature
        temp_match = re.search(r"Temperature: \(\s*([\de\+\-\.]+)\s*,\s*([\de\+\-\.]+)\s*,\s*([\de\+\-\.]+)\s*\)", timestep)
        if temp_match:
            temp = list(map(float, temp_match.groups()))
            temperature_data.append(temp)

        # Parse RMS Beam Size
        rms_beam_match = re.search(r"RMS Beam Size: \(\s*([\de\+\-\.]+)\s*,\s*([\de\+\-\.]+)\s*,\s*([\de\+\-\.]+)\s*\)", timestep)
        if rms_beam_match:
            rms_beam = list(map(float, rms_beam_match.groups()))
            rms_beam_size_data.append(rms_beam)

        # Parse (Normalized) RMS Emittance
        rms_emittance_match = re.search(r"(?:Normalized)? RMS Emittance: \s*([\de\+\-\.]+)\s*,\s*([\de\+\-\.]+)\s*,\s*([\de\+\-\.]+)\s*", timestep)
        if rms_emittance_match:
            rms_emittance = list(map(float, rms_emittance_match.groups()))
            rms_emittance_data.append(rms_emittance)

        # Parse L2-Norm of Temperature
        l2_temperature_match = re.search(r"L2-Norm of Temperature: ([\de\+\-\.]+)", timestep)
        if l2_temperature_match:
            l2_temperature = float(l2_temperature_match.group(1))
            l2_temperature_data.append(l2_temperature)

    return temperature_data, rms_beam_size_data, rms_emittance_data, l2_temperature_data

# Function to plot the data
# Function to plot the data
def plot_simulation_data(temperature_data, rms_beam_size_data, rms_emittance_data, l2_temperature_data, output_file):
    timesteps = list(range(1, len(temperature_data) + 1))

    temp_x = [temp[0] for temp in temperature_data]
    temp_y = [temp[1] for temp in temperature_data]
    temp_z = [temp[2] for temp in temperature_data]

    rms_x = [rms[0] for rms in rms_beam_size_data]
    rms_y = [rms[1] for rms in rms_beam_size_data]
    rms_z = [rms[2] for rms in rms_beam_size_data]

    rms_emittance_x = [emittance[0] for emittance in rms_emittance_data]
    rms_emittance_y = [emittance[1] for emittance in rms_emittance_data]
    rms_emittance_z = [emittance[2] for emittance in rms_emittance_data]

    plt.figure(figsize=(14, 14))

    # Plot Temperature
    plt.subplot(2, 2, 1)
    plt.plot(timesteps, temp_x, label='Temperature X')
    plt.plot(timesteps, temp_y, label='Temperature Y')
    plt.plot(timesteps, temp_z, label='Temperature Z')
    plt.xlabel('Timesteps')
    plt.ylabel('Temperature')
    plt.title('Temperature vs Timesteps')
    plt.legend()

    # Plot RMS Beam Size
    plt.subplot(2, 2, 2)
    plt.plot(timesteps, rms_x, label='RMS Beam Size X')
    plt.plot(timesteps, rms_y, label='RMS Beam Size Y')
    plt.plot(timesteps, rms_z, label='RMS Beam Size Z')
    plt.xlabel('Timesteps')
    plt.ylabel('RMS Beam Size')
    plt.title('RMS Beam Size vs Timesteps')
    plt.legend()

    # Plot (Normalized) RMS Emittance
    plt.subplot(2, 2, 3)
    plt.plot(timesteps, rms_emittance_x, label='RMS Emittance X')
    plt.plot(timesteps, rms_emittance_y, label='RMS Emittance Y')
    plt.plot(timesteps, rms_emittance_z, label='RMS Emittance Z')
    plt.xlabel('Timesteps')
    plt.ylabel('RMS Emittance')
    plt.title('(Normalized) RMS Emittance vs Timesteps')
    plt.legend()

    # Plot L2-Norm of Temperature
    plt.subplot(2, 2, 4)
    plt.plot(timesteps, l2_temperature_data, label='L2-Norm of Temperature')
    plt.xlabel('Timesteps')
    plt.ylabel('L2-Norm of Temperature')
    plt.title('L2-Norm of Temperature vs Timesteps')
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_file, format='pdf')
    plt.close()

# Path to the simulation output file
file_path = 'Heating2.log'

# Parse and plot the simulation data
temperature_data, rms_beam_size_data, c ,d = parse_simulation_data(file_path)
plot_simulation_data(temperature_data, rms_beam_size_data, c, d, output_file="SingleDIH2.pdf")

