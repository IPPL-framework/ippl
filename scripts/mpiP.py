import os
import sys
import re

def sum_send_bytes_from_file(file_path):
    total_bytes_sent = 0.0
    pattern = re.compile(
        r'^\s*Send\s+\d+\s+(\d+)\s+\d+\s+[\d.eE+-]+\s+[\d.eE+-]+\s+[\d.eE+-]+\s+([\d.eE+-]+)\s*$'
    )
    with open(file_path, 'r') as f:
        for line in f:
            match = pattern.match(line)
            if match:
                rank, sum_bytes = match.groups()
                if rank.isdigit():
                    total_bytes_sent += float(sum_bytes)
    return total_bytes_sent

if __name__ == '__main__':
    folder_path = sys.argv[1]
    if os.path.exists(folder_path):
        # Look for any file in the folder ending with .mpiP
        for filename in os.listdir(folder_path):
            if filename.endswith('.mpiP'):
                file_path = os.path.join(folder_path, filename)
                try:
                    total_bytes = sum_send_bytes_from_file(file_path)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                break  # Only one histogram file per run assumed
        print(f"Total bytes = ", total_bytes)
    else:
        print(f"Path given does not exist!")


