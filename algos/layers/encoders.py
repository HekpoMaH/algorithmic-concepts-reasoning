import struct
import torch
import numpy as np

def float2bitstring(f):
    [d] = struct.unpack('!i', struct.pack('!f', f))
    return f'{d:032b}'

def float2bin2(f):
    [d] = struct.unpack('>Q', struct.pack('>d', f))
    return f'{d:064b}'

def uint322bitstring(uint):
    return f'{uint:032b}'

def integer2bit(integer, num_bits=8):
    # Credit: https://github.com/KarenUllrich/pytorch-binary-converter/blob/master/binary_converter.py
    """Turn integer tensor to binary representation.
        Args:
            integer : torch.Tensor, tensor with integers
            num_bits : Number of bits to specify the precision. Default: 8.
        Returns:
            Tensor: Binary tensor. Adds last dimension to original tensor for
            bits.
    """
    dtype = integer.type()
    exponent_bits = -torch.arange(-(num_bits - 1), 1).type(dtype)
    exponent_bits = exponent_bits.repeat(integer.shape + (1,))
    out = integer.unsqueeze(-1) // 2 ** exponent_bits
    return ((out - (out % 1)) % 2).float()

def float2bit(f, dtype=torch.float32, num_bits=32):

    vwas = np.int32 if num_bits == 32 else np.int64
    as_ints = torch.tensor(f.cpu().numpy().view(vwas)).cuda()
    return integer2bit(as_ints, num_bits=num_bits).type(dtype)


class NumEncoder(torch.nn.Module):
    """ number to binary to feature space encoder. Defaults to 32 bits """
    def __init__(self, D_hidden, inp_type='uint', num_bits=32, bias=False):
        assert inp_type in ['uint', 'float']
        super(NumEncoder, self).__init__()
        self.lin = torch.nn.Linear(num_bits if inp_type == 'uint' else num_bits, D_hidden, bias=bias)
        self.num_bits = num_bits
        self.inp_type = inp_type

    def forward(self, inp):
        if self.inp_type == 'bits':
            bits = inp
        elif self.inp_type == 'uint':
            bits = integer2bit(inp, num_bits=self.num_bits)
        else:
            dtp = torch.float32 if self.num_bits == 32 else torch.float64
            bits = float2bit(inp, dtype=dtp, num_bits=self.num_bits)

        return self.lin(bits)

if __name__ == '__main__':
    print(integer2bit(torch.tensor(255), num_bits=9))
    exit(0)
    print(uint322bitstring(torch.tensor(320000)))
    print(integer2bit(torch.tensor([320000]), num_bits=32))
    print(float2bitstring(torch.tensor(1.)))
    print(float2bit(torch.tensor([1.])))
