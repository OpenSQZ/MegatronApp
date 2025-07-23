import argparse

parser = argparse.ArgumentParser(description="log file")
parser.add_argument('--file', type=str, required=True, help='file')

args = parser.parse_args()

log_file = args.file
send_start_times = {}
send_end_times = {}
recv_start_times = {}
recv_end_times = {}

with open(log_file, 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            if 'send start at' in line:
                parts = line.split('send start at')[1].split(',')
                timestamp = float(parts[0].strip())
                index = int(parts[1].strip())
                send_start_times[index] = timestamp
            if 'receive start at' in line:
                parts = line.split('receive start at')[1].split(',')
                timestamp = float(parts[0].strip())
                index = int(parts[1].strip())
                recv_start_times[index] = timestamp
            if 'send end at' in line:
                parts = line.split('send end at')[1].split(',')
                timestamp = float(parts[0].strip())
                index = int(parts[1].strip())
                send_end_times[index] = timestamp
            if 'receive end at' in line:
                parts = line.split('receive end at')[1].split(',')
                timestamp = float(parts[0].strip())
                index = int(parts[1].strip())
                recv_end_times[index] = timestamp
        except Exception as e:
            pass

send_delays = []
recv_delays = []
delays = []
for index in sorted(send_start_times.keys()):
    if index:
        send_delays.append(send_end_times[index] - send_start_times[index])
        recv_delays.append(recv_end_times[index] - recv_start_times[index])
        delays.append(min(send_end_times[index] - send_start_times[index], recv_end_times[index] - recv_start_times[index]))

if send_delays:
    avg_send_delay = sum(send_delays) / len(send_delays)
    print(f"avg send: {avg_send_delay:.6f}s")
if recv_delays:
    avg_recv_delay = sum(recv_delays) / len(recv_delays)
    print(f"avg recv: {avg_recv_delay:.6f}s")
if delays:
    avg_delay = sum(delays) / len(delays)
    print(f"avg delay: {avg_delay:.6f}s")
