import numpy as np
from torch_mlir.ir import Module, Context, WalkResult, WalkOrder, DenseResourceElementsAttr
from torch_mlir.dialects import torch as torch_d
from pathlib import Path
import migraphx


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

CONVERTERS = {}


def migraphx_converter(mlir_op, enabled: bool = True):

    def register_converter(fn):
        CONVERTERS[mlir_op] = fn
        return fn

    def disable_converter(fn):
        return fn

    if enabled:
        return register_converter
    else:
        return disable_converter


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


@migraphx_converter("torch.aten.convolution")
def torchmlir_convolution(mm, op, node_map):
    # torch.aten.convolution signature:
    #  convolution(Tensor input, Tensor weight, Tensor? bias, SymInt[] stride, SymInt[] padding, SymInt[] dilation, bool transposed, SymInt[] output_padding, SymInt groups)
    assert len(op.operands) == 9
    mgx_inputs = [node_map[i.get_name()] for i in op.operands]
    input, weight, bias = mgx_inputs[:3]
    stride, padding, dilation, transposed, output_padding, groups = mgx_inputs[3:]

    if transposed:
        raise RuntimeError("'transposed' parameter not supported.")

    if not all(i == 0 for i in output_padding):
        raise RuntimeError(
            "non-zero values for 'output_padding' not supported")

    out_mgx = mm.add_instruction(
        migraphx.op('convolution',
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    group=groups), [input, weight])

    out_shape = out_mgx.shape().lens()
    if bias:
        bias_mgx = mm.add_instruction(
            migraphx.op('broadcast', axis=1, out_lens=out_shape), [bias])
        out_mgx = mm.add_instruction(migraphx.op('add'), [out_mgx, bias_mgx])
    
    return out_mgx


class MLIRInterpreter:

    def __init__(self, module, sample_inputs=None) -> None:
        self.program = migraphx.program()
        self.mm = self.program.get_main_module()
        self.module = module
        self.dialect_resources = hacky_get_dialect_resources(module)

        self.IR_MAPPING = {}

    def process_op(self, op):
        print(op.name)
        print(type(op.opview))
        print(self.IR_MAPPING)

        result = WalkResult(WalkResult.ADVANCE)

        if op.name.startswith("func.func"):
            result = self.add_inputs(op)
        elif op.name.startswith("torch.constant"):
            result = self.add_constant(op)
        elif op.name.startswith("torch.vtensor.literal"):
            result = self.add_tensor_literal(op)
        elif op.name == "torch.prim.ListConstruct":
            result = self.add_list_construct(op)
        elif op.name == "func.return":
            result = self.add_return(op)
        else:
            result = self.convert_op(op)

        return result

    def convert_op(self, op):
        if not op.name in CONVERTERS:
            raise NotImplementedError(f"No converter found for {op.name}")

        mgx_ins = CONVERTERS[op.name](self.mm, op, self.IR_MAPPING)
        self.IR_MAPPING[op.result.get_name()] = mgx_ins
        return WalkResult(WalkResult.ADVANCE)

    def add_inputs(self, op):
        # in adding to func_op directly, arguments can be directly accessed
        # assert len(op.arguments) > 0

        # Not entirely sure when a program would have multiple regions and blocks
        for r in op.regions:
            for b in r.blocks:
                for arg in b.arguments:
                    # there is a "typeid" attribute but not sure how to use it properly atm
                    # for now resort to parsing a string for shape and type
                    arg_name = arg.get_name()
                    shape, dtype = type_parser(str(arg.type))
                    mgx_shape = migraphx.shape(lens=shape, type=dtype)
                    a_mgx = self.mm.add_parameter(arg_name, mgx_shape)
                    self.IR_MAPPING[arg_name] = a_mgx

        return WalkResult(WalkResult.ADVANCE)

    def add_constant(self, op):
        attr = None if len(op.attributes) == 0 else op.attributes[0].attr.value
        self.IR_MAPPING[op.result.get_name()] = attr
        return WalkResult(WalkResult.ADVANCE)

    def add_list_construct(self, op):
        # don't print lists: store them in a dict
        l = []
        for opd in op.operands:
            if opd.get_name() in self.IR_MAPPING.keys():
                l.append(self.IR_MAPPING[opd.get_name()])
            else:
                l.append(opd.get_name())
        self.IR_MAPPING[op.result.get_name()] = l
        return WalkResult(0)

    def add_tensor_literal(self, op):
        view = op.opview
        assert isinstance(view, torch_d.ValueTensorLiteralOp)
        value = DenseResourceElementsAttr(view.value)
        lt_index = str(value).find("<")
        gt_index = str(value).find(">")
        handle_name = str(value)[lt_index + 1:gt_index]

        #TODO: figure out how to parse this hex string in a numpy buffer
        tensor_hex_string = self.dialect_resources[handle_name]

        shape, dtype = type_parser(str(op.result.type))
        mgx_shape = migraphx.shape(lens=shape, type=dtype)

        ### TODO: remove
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
        if dtype in ["half_type", "float_type", "double_type"]:
            rand_literal = np.random.randn(*shape).astype(NP_TYPE_MAP[dtype])
        else:
            rand_literal = np.random.randint(0, size=shape).astype(
                NP_TYPE_MAP[dtype])
        ###

        mgx_lit = self.mm.add_literal(rand_literal)
        self.IR_MAPPING[op.result.get_name()] = mgx_lit

        return WalkResult(0)
    
    def add_return(self, op):
        outs = [self.IR_MAPPING[i.get_name()] for i in op.operands]
        self.mm.add_return(outs)
        return WalkResult(0)

    def run(self):
        assert len(self.module.body.operations) == 1
        func_op = self.module.body.operations[0]
        # self.add_inputs(func_op)
        func_op.walk(self.process_op, WalkOrder.PRE_ORDER)
        print(self.program)


if __name__ == "__main__":
    mlir_mod = load_module_from_path("./mlir/conv_2d_with_padding_static.mlir")
    interp = MLIRInterpreter(mlir_mod)
    interp.run()
    print(interp.program)
