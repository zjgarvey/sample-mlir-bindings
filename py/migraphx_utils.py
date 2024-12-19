import numpy as np
import struct
from torch_mlir.ir import Module, Context, WalkResult
from torch_mlir.dialects import torch as torch_d


def load_module_from_path(file_path) -> Module:
    # register the torch dialect with a context
    ctx = Context()
    torch_d.register_dialect(ctx)

    # parse the file contents as an mlir module
    with open(file_path, "r") as f:
        m = Module.parse(f.read(), ctx)
    return m


TYPE_MAP = {
    "i1": 'bool_type',
    "f16": 'half_type',
    "f32": 'float_type',
    "f64": 'double_type',
    "ui8": 'uint8_type',
    "si8": 'int8_type',
    "si16": 'int16_type',
    "si32": 'int32_type',
    "si64": 'int64_type'
}

NP_TYPE_MAP = {
    'bool_type': np.uint0,
    'half_type': np.half,
    'float_type': np.float32,
    'double_type': np.float64,
    'uint8_type': np.uint8,
    'int8_type': np.int8,
    'int16_type': np.int16,
    'int32_type': np.int32,
    'int64_type': np.int64,
}


PACK_CHAR_FROM_MGX_DTYPE = {
    'bool_type': "?",
    'half_type': "h",
    'float_type': "f",
    'double_type': "d",
    'uint8_type': "B",
    'int8_type': "b",
    'int16_type': "h",
    'int32_type': "i",
    'int64_type': "q",
}


def type_parser(type_str):
    assert type_str.startswith("!torch.vtensor")

    # For now assuming trye_str will ALWAYS be in the form "!torch.vtensor<[shape],dtype>"
    lens, mlir_type = type_str.strip("!torch.vtensor<>[").split("],")
    lens = [int(l) for l in lens.split(",")]
    mgx_type = TYPE_MAP[mlir_type]

    return lens, mgx_type


def hacky_get_dialect_resources(m: Module):
    m_lines = m.operation.get_asm().split("\n")
    dialect_resources = dict()
    key_idx_start = None
    key_idx_end = None
    for idx, l in enumerate(m_lines):
        if l == "{-#":
            key_idx_start = idx
        if l == "#-}":
            key_idx_end = idx
    assert key_idx_start is not None and key_idx_end is not None
    for line in m_lines[key_idx_start + 3:key_idx_end - 2]:
        line = line.lstrip(" ")
        key = line[0:line.find(":")]
        value = line[line.find('"') + 1:-1]
        dialect_resources[key] = value
    return dialect_resources


def decode_hex_string(hex_str, mgx_type, shape, from_resource_blob=True):
    assert mgx_type in NP_TYPE_MAP and mgx_type in PACK_CHAR_FROM_MGX_DTYPE
    
    start_idx = 10 if from_resource_blob else 2
    buffer = bytes.fromhex(hex_str[start_idx:])
    num_elem = np.prod(shape) if len(shape) > 0 else np.array([1])
    value_tuple = struct.unpack(PACK_CHAR_FROM_MGX_DTYPE[mgx_type] * num_elem, buffer)
    return np.array(value_tuple, dtype=NP_TYPE_MAP[mgx_type]).reshape(shape)