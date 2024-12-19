import migraphx


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
