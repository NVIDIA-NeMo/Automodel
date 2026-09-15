# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Repository-specific Import Linter contracts."""

from __future__ import annotations

from pathlib import Path

from grimp import ImportGraph
from importlinter import Contract, ContractCheck, fields, output
from importlinter.domain.helpers import module_expressions_to_modules

from tools.component_imports import ComponentImport, find_component_imports


class ComponentInterfaceContract(Contract):
    """Require every component consumer to use exported interfaces."""

    type_name = "component_interface"
    modules = fields.ListField(subfield=fields.ModuleExpressionField())

    def check(self, graph: ImportGraph, verbose: bool) -> ContractCheck:
        """Reject component imports that bypass package exports."""
        modules = {
            module.name
            for module in module_expressions_to_modules(graph, self.modules)  # type: ignore[arg-type]
        }
        component_imports = find_component_imports(_PROJECT_ROOT, modules)
        violations = [component_import for component_import in component_imports if not component_import.is_public]
        return ContractCheck(kept=not violations, metadata={"component_interface_violations": violations})

    def render_broken_contract(self, check: ContractCheck) -> None:
        """Render public-interface violations."""
        violations: list[ComponentImport] = check.metadata["component_interface_violations"]
        output.print_error("Component imports must use exported symbols:", bold=False)
        output.new_line()
        for violation in violations:
            output.print_error(
                f"- {violation.path}:{violation.line_number}: {violation.violation}",
                bold=False,
            )
            output.new_line()
        output.new_line()


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
