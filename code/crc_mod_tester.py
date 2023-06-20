import crcmod
from bitarray import bitarray

# Define the polynomial (x^16 + x^12 + x^5 + 1)
poly = 0x11021

# Create a CRC object with the specified polynomial and parameters
crc_obj = crcmod.predefined.Crc('crc-16', poly, initCrc=0xFFFF, xorOut=0xFFFF, rev=False)

# Generate a message to compute the CRC code for
message = bytearray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Convert the message to a bit sequence
bit_seq = bitarray()
bit_seq.frombytes(message)

# Calculate the CRC code for the bit sequence using the CRC object
crc_code_python = crc_obj.new(bit_seq).digest()

# Display the CRC code in hexadecimal format
print(crc_code_python.hex())