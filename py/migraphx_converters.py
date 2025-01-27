import numpy as np
import migraphx
from migraphx_utils import NP_TYPE_MAP

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


def broadcast_for_elemwise_op(mgx_module,
                              inp,
                              other):
    assert isinstance(inp, migraphx.instruction_ref)
    assert isinstance(other, migraphx.instruction_ref)

    if (inp == other):
        return inp, other

    inp_shape = inp.shape().lens()
    other_shape = other.shape().lens()

    out_shape = np.broadcast_shapes(inp_shape, other_shape)
    if len(out_shape) == 0 or inp_shape == other_shape:
        return inp, other

    inp = mgx_module.add_instruction(
        migraphx.op('multibroadcast', out_lens=list(out_shape)), [inp])

    other = mgx_module.add_instruction(
        migraphx.op('multibroadcast', out_lens=list(out_shape)), [other])

    return inp, other


@migraphx_converter("torch.aten.convolution")
def torchmlir_convolution(mm, op, node_map):
    # torch.aten.convolution signature:
    #  convolution(Tensor input, Tensor weight, Tensor? bias, SymInt[] stride, SymInt[] padding, SymInt[] dilation, bool transposed, SymInt[] output_padding, SymInt groups)
    assert len(op.operands) == 9
    mgx_inputs = [node_map[i.get_name()] for i in op.operands]
    input, weight, bias = mgx_inputs[:3]
    stride, padding, dilation, transposed, output_padding, groups = mgx_inputs[
        3:]

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


@migraphx_converter("torch.aten.add.Scalar")
@migraphx_converter("torch.aten.add.Tensor")
def torchmlir_add(mm, op, node_map):
    # torch.aten.add.Scalar signature:
    # aten::add.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor
    # torch.aten.add.Tensor signature:
    # aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
    assert len(op.operands) == 3, "Expected 3 operands: input, other, alpha"
    mgx_inputs = [node_map[i.get_name()] for i in op.operands]
    input, other, alpha = mgx_inputs
    assert isinstance(input, migraphx.instruction_ref), "Expected input to torch.aten.add.Scalar to be a tensor"

    if not isinstance(other, migraphx.instruction_ref):
        other *= alpha
        other = mm.add_literal(np.array(other).astype(NP_TYPE_MAP[input.shape().type_string()]))
    elif alpha != 1:
        alpha = mm.add_literal(np.array(alpha).astype(NP_TYPE_MAP[input.shape().type_string()]))
        other = mm.add_insturction(migraphx.op("mul"), [alpha, other])

    input, other = broadcast_for_elemwise_op(mm, input, other)

    return mm.add_instruction(migraphx.op('add'), [input, other])


@migraphx_converter("torch.aten.sub.Scalar")
@migraphx_converter("torch.aten.sub.Tensor")
def torchmlir_sub(mm, op, node_map):
    # torch.aten.sub.Scalar signature:
    # aten::sub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor
    # torch.aten.sub.Tensor signature:
    # aten::sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
    assert len(op.operands) == 3, "Expected 3 operands: input, other, alpha"
    mgx_inputs = [node_map[i.get_name()] for i in op.operands]
    input, other, alpha = mgx_inputs
    assert isinstance(input, migraphx.instruction_ref), "Expected input to torch.aten.add.Scalar to be a tensor"

    if not isinstance(other, migraphx.instruction_ref):
        other *= alpha
        other = mm.add_literal(np.array(other).astype(NP_TYPE_MAP[input.shape().type_string()]))
    elif alpha != 1:
        alpha = mm.add_literal(np.array(alpha).astype(NP_TYPE_MAP[input.shape().type_string()]))
        other = mm.add_insturction(migraphx.op("mul"), [alpha, other])

    input, other = broadcast_for_elemwise_op(mm, input, other)

    return mm.add_instruction(migraphx.op('sub'), [input, other])


@migraphx_converter("torch.aten.sqrt")
def torchmlir_sqrt(mm, op, node_map):
    # torch.aten.sqrt signature: aten::sqrt(Tensor self) -> Tensor
    assert len(op.operands) == 1, "Expected 1 operand for torch.aten.sqrt"
    mgx_inputs = [node_map[i.get_name()] for i in op.operands]
    input = mgx_inputs[0]

    return mm.add_instruction(migraphx.op('sqrt'), [input])

@migraphx_converter("torch.aten.reciprocal")
def torchmlir_recip(mm, op, node_map):
    # torch.aten.reciprocal signature: torch.aten.reciprocal(Tensor self) -> Tensor
    assert len(op.operands) == 1, "Expected 1 operand for torch.aten.sqrt"
    mgx_inputs = [node_map[i.get_name()] for i in op.operands]
    input = mgx_inputs[0]

    return mm.add_instruction(migraphx.op('recip'), [input])


@migraphx_converter("torch.aten.mul.Scalar")
@migraphx_converter("torch.aten.mul.Tensor")
def torchmlir_mul(mm, op, node_map):
    # torch.aten.mul.Scalar signature: aten::mul.Scalar(Tensor self, Scalar other) -> Tensor
    # torch.aten.mul.Tensor signature: aten::mul.Tensor(Tensor self, Tensor other) -> Tensor
    assert len(op.operands) == 2, "Expected 2 operands: input, other"
    mgx_inputs = [node_map[i.get_name()] for i in op.operands]
    input, other = mgx_inputs
    assert isinstance(input, migraphx.instruction_ref)

    if not isinstance(other, migraphx.instruction_ref):
        other = mm.add_literal(np.array(other).astype(NP_TYPE_MAP[input.shape().type_string()]))

    input, other = broadcast_for_elemwise_op(mm, input, other)

    return mm.add_instruction(migraphx.op('mul'), [input, other])


@migraphx_converter("torch.aten.div.Scalar")
@migraphx_converter("torch.aten.div.Tensor")
def torchmlir_div(mm, op, node_map):
    # torch.aten.div.Scalar signature: aten::div.Scalar(Tensor self, Scalar other) -> Tensor
    # torch.aten.div.Tensor signature: aten::div.Tensor(Tensor self, Tensor other) -> Tensor
    assert len(op.operands) == 2, "Expected 2 operands: input, other"
    mgx_inputs = [node_map[i.get_name()] for i in op.operands]
    input, other = mgx_inputs
    assert isinstance(input, migraphx.instruction_ref)

    if not isinstance(other, migraphx.instruction_ref):
        other = mm.add_literal(np.array(other).astype(NP_TYPE_MAP[input.shape().type_string()]))

    input, other = broadcast_for_elemwise_op(mm, input, other)

    return mm.add_instruction(migraphx.op('div'), [input, other])


@migraphx_converter("torch.aten.unsqueeze")
def torchmlir_unsqueeze(mm, op, node_map):
    # torch.aten.unsqueeze signature: aten::unsqueeze(Tensor(a) self, int dim) -> Tensor(a)
    assert len(op.operands) == 2, "Expected 2 operands for torch.aten.unsqueeze: input, dim"
    mgx_inputs = [node_map[i.get_name()] for i in op.operands]
    input, dim = mgx_inputs

    return mm.add_instruction(migraphx.op('unsqueeze', axes=[dim]), [input])


@migraphx_converter("torch.aten.relu")
def torchmlir_unsqueeze(mm, op, node_map):
    # torch.aten.relu signature: aten::relu(Tensor self) -> Tensor
    assert len(op.operands) == 1, "Expected 1 operand for torch.aten.relu"
    mgx_inputs = [node_map[i.get_name()] for i in op.operands]
    input = mgx_inputs[0]

    return mm.add_instruction(migraphx.op('relu'), [input])


@migraphx_converter("torch.aten.max_pool2d")
def torchmlir_max_pool2d(mm, op, node_map):
    # torch.aten.max_poll2d signature:
    #  aten::max_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> Tensor
    assert len(op.operands) == 6
    mgx_inputs = [node_map[i.get_name()] for i in op.operands]
    input, kernel_size, stride, padding, dilation, ceil_mode = mgx_inputs

    if not all(i == 1 for i in dilation):
        raise RuntimeError('Dilations are currently not supported.')
    
    mode = migraphx.op.pooling_mode.max
    mode = int(mode) if not hasattr(mode, '__index__') else mode

    return mm.add_instruction(
        migraphx.op('pooling',
                    mode=mode,
                    padding=padding,
                    stride=stride,
                    lengths=kernel_size,
                    ceil_mode=ceil_mode), [input])


@migraphx_converter("torch.aten.sum.dim_IntList")
def torchmlir_sum(mm, op, node_map):
    # aten::sum.dim_IntList(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
    assert len(op.operands) == 4
    mgx_inputs = [node_map[i.get_name()] for i in op.operands]
    input, dims, keepdim, dtype = mgx_inputs

    assert dtype is None, "dtype not supported yet"

    sum_ = mm.add_instruction(migraphx.op('reduce_sum', axes=dims), [input])

    if not keepdim:
        sum_ = mm.add_instruction(migraphx.op('squeeze', axes=dims), [sum_])
    
    return sum_


@migraphx_converter("torch.aten.view")
def torchmlir_view(mm, op, node_map):
    # aten::view(Tensor(a) self, SymInt[] size) -> Tensor(a)
    assert len(op.operands) == 2, "Expected 2 operands for torch.aten.view: input, size"
    mgx_inputs = [node_map[i.get_name()] for i in op.operands]
    input, size = mgx_inputs

    return mm.add_instruction(migraphx.op('reshape', dims=size), [input])


@migraphx_converter("torch.aten.transpose.int")
def torchmlir_transpose(mm, op, node_map):
    # aten::transpose.int(Tensor(a) self, int dim0, int dim1) -> Tensor(a)
    assert len(op.operands) == 3, "Expected 3 operands for torch.aten.transpose.int: input, dim0, dim1"
    mgx_inputs = [node_map[i.get_name()] for i in op.operands]
    input, dim0, dim1 = mgx_inputs
    perm = list(range(len(input.shape().lens())))
    perm[dim0] = dim1
    perm[dim1] = dim0

    return mm.add_instruction(migraphx.op('transpose', permutation=perm), [input])


@migraphx_converter("torch.aten.mm")
def torchmlir_mm(mm, op, node_map):
    # aten::mm(Tensor self, Tensor mat2) -> Tensor
    assert len(op.operands) == 2, "Expected 2 operands for torch.aten.mm: input, mat2"
    mgx_inputs = [node_map[i.get_name()] for i in op.operands]
    mat1, mat2 = mgx_inputs

    return mm.add_instruction(migraphx.op('dot'), [mat1, mat2])
