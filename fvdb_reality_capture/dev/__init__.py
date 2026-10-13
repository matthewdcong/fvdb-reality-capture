# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
from importlib import import_module
from types import ModuleType

__all__ = ["s3"]


def __getattr__(name: str) -> ModuleType:
    # S3 dependencies belong to the dev extra and are not needed by frgs.
    if name == "s3":
        module = import_module(".s3", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
