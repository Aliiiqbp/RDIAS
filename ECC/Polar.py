from python_polar_coding.channels import SimpleBPSKModulationAWGN
from python_polar_coding.polar_codes import FastSSCPolarCodec
from python_polar_coding.simulation.functions import (
    compute_fails,
    generate_binary_message,
)

N = 256
K = 64
design_snr = 0.0
messages = 1000
# SNR in [.0, .5, ..., 4.5, 5]
snr_range = [i / 2 for i in range(11)]

codec = FastSSCPolarCodec(N=N, K=K, design_snr=design_snr)
bpsk = SimpleBPSKModulationAWGN(fec_rate=K/N)

result_ber = dict()
result_fer = dict()

print('Python polar coding simulation')
print(f'Simulating ({codec.N}, {codec.K}) systematic polar code with Design SNR {codec.design_snr} dB')
print()
print('\tSNR (dB)|\tBER\t|\tFER')

for snr in snr_range:
    ber = 0
    fer = 0

    for _ in range(messages):
        msg = generate_binary_message(size=K)
        print('meg', msg)
        encoded = codec.encode(msg)
        print('enc', encoded)
        transmitted = bpsk.transmit(message=encoded, snr_db=snr)
        print('trans', transmitted)
        decoded = codec.decode(transmitted)
        print('dec', decoded)
        if msg.all() == decoded.all():
            print('YYY')
        else:
            print('NNN')

        bit_errors, frame_error = compute_fails(msg, decoded)
        ber += bit_errors
        fer += frame_error

    result_ber[snr] = ber / (messages * codec.K)
    result_fer[snr] = fer / messages

    # print(f'\t{snr}\t|\t{result_ber[snr]:.4f}\t|\t{result_fer[snr]:.4f}')
