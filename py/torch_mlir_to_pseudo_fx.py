from torch_mlir.ir import Module, Context, WalkResult
from torch_mlir.dialects import torch as torch_d
from pathlib import Path
from typing import Dict, Any, List, Union
import argparse


def load_module_from_path(file_path: Union[Path, str]) -> Module:
    # register the torch dialect with a context
    ctx = Context()
    torch_d.register_dialect(ctx)

    # parse the file contents as an mlir module
    with open(file_path, "r") as f:
        m = Module.parse(f.read(), ctx)
    return m


def get_pseudo_fx_graph(m: Module) -> List[str]:
    # get the func op
    func_op = m.body.operations[0]
    instruction_strings: List[str] = []

    # print(f'func op info:')
    # for attribute in func_op.attributes:
    #     print(f'\t{attribute.name} : {attribute.attr}')
    func_name = func_op.attributes["sym_name"].value
    instruction_strings.append(f"{func_name}():")
    for r in func_op.regions:
        for b in r.blocks:
            for arg in b.arguments:
                # print(f'\targument name : {arg.get_name()} with type : {arg.type}')
                num_uses = 0
                for use in arg.uses:
                    num_uses += 1
                    # print(f'\t\t{arg.get_name()} is used by: {use.owner.results[0].get_name()}')
                instruction_strings.append(
                    f'\t{arg.get_name()} : [num_users={num_uses}] = placeholder[target={arg.get_name().lstrip("%")}][type_expr={arg.type}]'
                )

    # for storing references to constants and lists in the IR
    constant_associator: Dict[str, Any] = dict()
    list_associator: Dict[str, List] = dict()

    def process_op(op):
        """This function will look at a torch-mlir op and try to generate an instruction string in a pseudo fx_graph style"""
        if op.name == "func.func":
            # do nothing
            return WalkResult(0)
        if op.name == "func.return":
            instruction_strings.append(
                f'\treturn {op.operands[0].get_name().lstrip("%")}'
            )
            return WalkResult(0)
        result = op.result
        if op.name.startswith("torch.constant"):
            # don't print constants: store them in a dict
            attr = None if len(op.attributes) == 0 else op.attributes[0].attr.value
            constant_associator[result.get_name()] = attr
            return WalkResult(0)
        if op.name == "torch.prim.ListConstruct":
            # don't print lists: store them in a dict
            l = []
            for opd in op.operands:
                if opd.get_name() in constant_associator.keys():
                    l.append(constant_associator[opd.get_name()])
                else:
                    l.append(opd.get_name())
            list_associator[result.get_name()] = l
            return WalkResult(0)

        result_type = result.type
        # TODO: find something that converts IR types back to torch types
        result_name = result.get_name()
        # get the "args" for the op
        inputs = []
        for opd in op.operands:
            opd_name = opd.get_name()
            if opd_name in constant_associator.keys():
                inputs.append(constant_associator[opd_name])
                continue
            if opd_name in list_associator.keys():
                inputs.append(tuple(list_associator[opd_name]))
                continue
            inputs.append(opd_name)
        # make a string from args:
        arg_str = "("
        for i in inputs:
            arg_str += f"{i}, "
        arg_str += ")"
        # get number of uses:
        num_uses = 0
        for use in result.uses:
            num_uses += 1
        # get keyword args (attributes ?)
        attrs = dict()
        for attribute in op.attributes:
            attrs[attribute.name] = attribute.attr
        # generate the fake instruction string
        instruction_strings.append(
            f"\t{result_name} : [num_users = {num_uses}] = call_function[target={op.name}](args={arg_str}, kwargs={attrs})[type_expr={result_type}]"
        )
        return WalkResult(0)

    func_op.walk(process_op)
    return instruction_strings


def main(args):
    m = load_module_from_path(args.input_file)
    pseudo_fx = get_pseudo_fx_graph(m)
    f = None
    if args.output_file is not None:
        f = open(args.output_file, "w")
    print(*pseudo_fx, sep="\n", file=f)
    if f is not None:
        f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("torch_mlir_to_pseudo_fx.py")
    parser.add_argument("input_file", type=str)
    parser.add_argument("-o", "--output_file", type=str)
    args = parser.parse_args()
    main(args)
