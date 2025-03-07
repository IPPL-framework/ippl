import re
import matplotlib.pyplot as plt

# Define the parameter choices and corresponding file names
parameter_choices = [1, 2, 3, 4, 5, 6, 8]
file_names = [f'DistHeating{param}.err' for param in parameter_choices]

# Initialize dictionaries to hold the data
rms_beam_size_x = {param: [] for param in parameter_choices}
rms_emittance_x = {param: [] for param in parameter_choices}
temperature_x_dict = {param: [] for param in parameter_choices}
l2_norm_temp = {param: [] for param in parameter_choices}

# Function to parse data from a single file
def parse_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    # Split content into timesteps
    timesteps = re.split(r'LeapFrog Step \d+ Finished\.', content)
    for timestep in timesteps:
        # Find RMS beam size x
        beam_size_match = re.search(r"RMS Beam Size:\s*([\de\+\-\.]+)\s*,\s*([\de\+\-\.]+)\s*,\s*([\de\+\-\.]+)", timestep)
        if beam_size_match:
            beam_size_x = float(beam_size_match.group(1))
        else:
            beam_size_x = None

        # Find RMS emittance x
        emittance_match = re.search(r"RMS Emittance:\s*([\de\+\-\.]+)\s*,\s*([\de\+\-\.]+)\s*,\s*([\de\+\-\.]+)", timestep)
        if emittance_match:
            emittance_x = float(emittance_match.group(1))
        else:
            emittance_x = None

        # Find Temperature x
        temp_match = re.search(r"Temperature:\s*\(\s*([\de\+\-\.]+)\s*,\s*([\de\+\-\.]+)\s*,\s*([\de\+\-\.]+)\s*\)", timestep)
        if temp_match:
            temperature_x = float(temp_match.group(1))
        else:
            temperature_x = None

        # Find L2 norm of temperature
        l2_temp_match = re.search(r"L2-Norm of Temperature:\s*([\de\+\-\.]+)", timestep)
        if l2_temp_match:
            l2_temp = float(l2_temp_match.group(1))
        else:
            l2_temp = None

        yield beam_size_x, emittance_x, temperature_x, l2_temp

# Parse each file and collect data
for param, file_name in zip(parameter_choices, file_names):
    for beam_size_x, emittance_x, temperature_x, l2_temp in parse_file(file_name):
        if beam_size_x is not None:
            rms_beam_size_x[param].append(beam_size_x)
        if emittance_x is not None:
            rms_emittance_x[param].append(emittance_x)
        if temperature_x is not None:
            temperature_x_dict[param].append(temperature_x)
        if l2_temp is not None:
            l2_norm_temp[param].append(l2_temp)

# Plot the data
plt.figure(figsize=(14, 10))

# Plot RMS Beam Size X
plt.subplot(2, 2, 1)
for param in parameter_choices:
    plt.plot(range(len(rms_beam_size_x[param])), rms_beam_size_x[param], label=f'r_cut= {param}h')
plt.xlabel('Timesteps')
plt.ylabel('RMS Beam Size X')
plt.title('RMS Beam Size X vs Timesteps')
plt.legend()
plt.grid(True)

# Plot RMS Emittance X
plt.subplot(2, 2, 2)
for param in parameter_choices:
    plt.plot(range(len(rms_emittance_x[param])), rms_emittance_x[param], label=f'r_cut= {param}h')
plt.xlabel('Timesteps')
plt.ylabel('RMS Emittance X')
plt.title('RMS Emittance X vs Timesteps')
plt.legend()
plt.grid(True)

# Plot Temperature X
plt.subplot(2, 2, 3)
for param in parameter_choices:
    plt.plot(range(len(temperature_x_dict[param])), temperature_x_dict[param], label=f'r_cut= {param}h')
plt.xlabel('Timesteps')
plt.ylabel('Temperature X')
plt.title('Temperature X vs Timesteps')
plt.legend()
plt.grid(True)

# Plot L2-Norm of Temperature
plt.subplot(2, 2, 4)
for param in parameter_choices:
    plt.plot(range(len(l2_norm_temp[param])), l2_norm_temp[param], label=f'r_cut= {param}h')
plt.xlabel('Timesteps')
plt.ylabel('L2-Norm of Temperature')
plt.title('L2-Norm of Temperature vs Timesteps')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('DistHeating_simresults_h1-8.pdf', format='pdf')
plt.close()  # Close the plot to avoid displaying it in interactive environments

