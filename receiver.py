from audio_processing import *
from utils import binary_to_string


def main():
    # Build the known pad waveform used to mark the start/end of the message
    pad_wave = get_pad_wave()
    # Record raw audio from the microphone for LISTEN_DURATION seconds
    rec = record_audio()

    # Clean up the recording by filtering out-of-band noise (optional)
    # rec = bandpass_filter(rec, FREQ_MIN, FREQ_MAX, SAMPLE_RATE)

    # Display the spectrogram of the entire recording
    plot_spectrogram(rec, "Full Recording")

    # Find the sample indices where the pad waveform occurs in the recording
    (p1_samp, _), (p2_samp, _) = find_time_delays(pad_wave, rec)
    print(f"Pad #1 at {p1_samp}, Pad #2 at {p2_samp}")

    # Extract the section of audio between the two pad occurrences (the actual message)
    pad_len = len(pad_wave)
    msg_sig = rec[int(p1_samp + pad_len): int(p2_samp)]

    # Convert the message audio into a string of bits
    bit_string = process_audio_to_bits(msg_sig)
    print("Bit sequence:", bit_string)

    # Decode the bit string into readable text
    text = binary_to_string(bit_string)
    print("Decoded text:", repr(text))

    # Display the spectrogram of just the extracted message
    plot_spectrogram(msg_sig, "Message Only")


if __name__ == "__main__":
    main()
