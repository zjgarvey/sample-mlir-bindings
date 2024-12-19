from torch_mlir.ir import Module, Context, WalkResult, WalkOrder, DenseResourceElementsAttr
from torch_mlir.dialects import torch as torch_d
import migraphx

from migraphx_converters import CONVERTERS
from migraphx_utils import hacky_get_dialect_resources, type_parser, decode_hex_string


class MLIRInterpreter:

    def __init__(self, module, sample_inputs=None) -> None:
        self.program = migraphx.program()
        self.mm = self.program.get_main_module()
        self.module = module
        self.dialect_resources = hacky_get_dialect_resources(module)

        self.IR_MAPPING = {}

    def process_op(self, op):
        result = WalkResult(WalkResult.INTERRUPT)

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
        return WalkResult(WalkResult.ADVANCE)

    def add_tensor_literal(self, op):
        view = op.opview
        assert isinstance(view, torch_d.ValueTensorLiteralOp)
        value = DenseResourceElementsAttr(view.value)
        lt_index = str(value).find("<")
        gt_index = str(value).find(">")
        handle_name = str(value)[lt_index + 1:gt_index]

        tensor_hex_string = self.dialect_resources[handle_name]

        shape, dtype = type_parser(str(op.result.type))
        # mgx_shape = migraphx.shape(lens=shape, type=dtype)

        lit = decode_hex_string(tensor_hex_string, dtype, shape)

        mgx_lit = self.mm.add_literal(lit)
        self.IR_MAPPING[op.result.get_name()] = mgx_lit

        return WalkResult(WalkResult.ADVANCE)

    def add_return(self, op):
        outs = [self.IR_MAPPING[i.get_name()] for i in op.operands]
        self.mm.add_return(outs)
        return WalkResult(WalkResult.ADVANCE)

    def run(self):
        assert len(self.module.body.operations) == 1
        func_op = self.module.body.operations[0]
        # self.add_inputs(func_op)
        func_op.walk(self.process_op, WalkOrder.PRE_ORDER)
        print(self.program)