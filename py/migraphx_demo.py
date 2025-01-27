import numpy as np
from migraphx_interpreter import MLIRInterpreter
from migraphx_utils import load_module_from_path
import migraphx

if __name__ == "__main__":
    mlir_mod = load_module_from_path("../mlir/resnet50.mlir")
    interp = MLIRInterpreter(mlir_mod)
    interp.run()
    print(interp.program)

    p = interp.program
    p.compile(migraphx.get_target('gpu'))
    
    inputs = np.load("../artifacts/inputs/resnet50.npy")
    gold_outputs = np.load("../artifacts/outputs/resnet50.npy")

    mgx_outputs = np.array(p.run({p.get_parameter_names()[0]: inputs})[0]).reshape(gold_outputs.shape)

    assert np.allclose(gold_outputs, mgx_outputs, rtol=1e-3, atol=1e-3), "Output mismatch"
    print("Outputs Match!")
