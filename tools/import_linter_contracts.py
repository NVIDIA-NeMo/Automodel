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
from importlinter import ContractCheck, output
from importlinter.contracts.independence import IndependenceContract
from importlinter.domain.helpers import module_expressions_to_modules

from tools.component_imports import ComponentImport, find_component_imports


class ComponentInterfaceContract(IndependenceContract):
    """Require every component consumer to use exported interfaces."""

    type_name = "component_interface"

    # Import Linter discovers fields on the concrete class rather than base classes.
    modules = IndependenceContract.modules
    ignore_imports = IndependenceContract.ignore_imports
    unmatched_ignore_imports_alerting = IndependenceContract.unmatched_ignore_imports_alerting

    def check(self, graph: ImportGraph, verbose: bool) -> ContractCheck:
        """Allow direct component edges only when they use package exports."""
        modules = {
            module.name
            for module in module_expressions_to_modules(graph, self.modules)  # type: ignore[arg-type]
        }
        component_imports = find_component_imports(_PROJECT_ROOT, modules)

        for component_import in component_imports:
            self._remove_direct_import(graph, component_import)

        independence_check = super().check(graph, verbose)
        violations = [component_import for component_import in component_imports if not component_import.is_public]
        independence_check.kept = independence_check.kept and not violations
        independence_check.metadata["component_interface_violations"] = violations
        return independence_check

    def render_broken_contract(self, check: ContractCheck) -> None:
        """Render public-interface violations before any remaining dependency chains."""
        violations: list[ComponentImport] = check.metadata["component_interface_violations"]
        if violations:
            output.print_error("Component imports must use exported symbols:", bold=False)
            output.new_line()
            for violation in violations:
                output.print_error(
                    f"- {violation.path}:{violation.line_number}: {violation.violation}",
                    bold=False,
                )
                output.new_line()
            output.new_line()

        if check.metadata["invalid_chains"]:
            super().render_broken_contract(check)

    @staticmethod
    def _remove_direct_import(graph: ImportGraph, component_import: ComponentImport) -> None:
        if component_import.importer not in graph.modules:
            return
        imported_modules = graph.find_modules_directly_imported_by(component_import.importer)
        for imported_module in imported_modules:
            if not (
                imported_module == component_import.target_component
                or imported_module.startswith(f"{component_import.target_component}.")
            ):
                continue
            import_details = graph.get_import_details(
                importer=component_import.importer,
                imported=imported_module,
            )
            if any(detail["line_number"] == component_import.line_number for detail in import_details):
                graph.remove_import(importer=component_import.importer, imported=imported_module)


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
