import subprocess
import argparse
from pathlib import Path
import logging
from typing import List

logger = logging.getLogger(__name__)
PARENT_DIR = Path(__file__).parents[1]


class IreeCompileException(RuntimeError):
    pass


class IreeRunException(RuntimeError):
    pass


GENERIC_COMPILE_ARGS = {
    "cpu": ["--iree-hal-target-backends=llvm-cpu", "--iree-llvmcpu-target-cpu=host"],
    "Mi300x": ["--iree-hal-target-backends=rocm", "--iree-hip-target=gfx942"],
}

GENERIC_RUN_MODULE_ARGS = {
    "cpu": ["--device=local-task"],
    "Mi300x": ["--device=hip"],
}


def compile(mlir_path: Path, base_compile_args: List[str]):
    vmfb_path = PARENT_DIR / "artifacts" / "vmfb" / (mlir_path.stem + ".vmfb")

    compile_args = base_compile_args + [
        str(mlir_path.resolve()),
        "-o",
        str(vmfb_path.resolve()),
    ]
    compile_cmd = subprocess.list2cmdline(compile_args)
    logger.info(
        f"Launching compile command:\n"  #
        f"{compile_cmd}"
    )
    compile_ret = subprocess.run(
        compile_cmd, shell=True, capture_output=True, cwd=PARENT_DIR
    )
    if compile_ret.returncode != 0:
        logger.error(f"Compilation of '{mlir_path}' failed")
        logger.error("iree-compile stdout:")
        logger.error(compile_ret.stdout.decode("utf-8"))
        logger.error("iree-compile stderr:")
        logger.error(compile_ret.stderr.decode("utf-8"))
        raise IreeCompileException(f"  '{mlir_path.name}' compile failed")

    return vmfb_path


def run(vmfb_path: Path, base_run_module_args: List[str], compare: bool):
    input_files = [
        thing
        for thing in Path.glob(
            PARENT_DIR / "artifacts" / "inputs", vmfb_path.stem + "*"
        )
    ]
    torch_outputs = [
        thing
        for thing in Path.glob(
            PARENT_DIR / "artifacts" / "outputs", vmfb_path.stem + "_torch*"
        )
    ]
    iree_outputs = [
        PARENT_DIR / "artifacts" / "outputs" / f"{vmfb_path.stem}_iree_{idx}.npy"
        for (idx, _) in enumerate(torch_outputs)
    ]
    run_module_args = base_run_module_args + [f"--module={vmfb_path.resolve()}"]
    run_module_args += [f"--input=@{input.resolve()}" for input in input_files]
    run_module_args += [f"--output=@{output.resolve()}" for output in iree_outputs]
    if compare:
        run_module_args += [
            f"--expected_output=@{torch_out.resolve()}" for torch_out in torch_outputs
        ]
    run_module_cmd = subprocess.list2cmdline(run_module_args)
    logger.info(
        f"Launching run_module command:\n"  #
        f"{run_module_cmd}"
    )
    run_module_ret = subprocess.run(
        run_module_cmd, shell=True, capture_output=True, cwd=PARENT_DIR
    )
    if run_module_ret.returncode != 0:
        logger.error(f"Run of '{vmfb_path}' failed")
        logger.error("iree-run-module stdout:")
        logger.error(run_module_ret.stdout.decode("utf-8"))
        logger.error("iree-run-module stderr:")
        logger.error(run_module_ret.stderr.decode("utf-8"))
        raise IreeRunException(f"  '{vmfb_path.name}' run failed")
    return run_module_ret.stdout.decode("utf-8")


def main(args):
    log_level = logging.INFO if args.verbose else logging.ERROR
    if args.log_file:
        logging.basicConfig(filename=args.log_file, level=log_level)
    mlir_files = Path.glob(PARENT_DIR / "mlir", "*.mlir")
    base_compile_args = ["iree-compile"] + GENERIC_COMPILE_ARGS[args.device]
    base_run_module_args = ["iree-run-module"] + GENERIC_RUN_MODULE_ARGS[args.device]
    for m in mlir_files:
        vmfb_path = compile(m, base_compile_args)
        std_out = run(vmfb_path, base_run_module_args, args.compare)
        print(f"Result for {m.stem}:")
        print(std_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--device",
        choices=["cpu", "Mi300x"],
        required=True,
    )
    parser.add_argument(
        "-c",
        "--compare",
        help="compare iree result with torch result",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="sets the logging level to INFO",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "-l",
        "--log-file",
        help="where to store logger output",
        type=str,
    )
    main(parser.parse_args())
