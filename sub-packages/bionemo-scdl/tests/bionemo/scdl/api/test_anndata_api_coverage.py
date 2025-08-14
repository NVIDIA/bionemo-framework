#!/usr/bin/env python3
"""
AnnData API Coverage Tool (usage and mirror modes)

This tool can analyze Python files to:
 1) usage mode: detect which parts of the AnnData API a codebase USES
 2) mirror mode: detect which parts of the AnnData API a class/module MIRRORS

Mirror mode is the default, intended to check AnnData API surface parity for
re-implementations (e.g., a dataset class that mirrors AnnData attributes and
methods with a different backing store).

Examples:
  # Mirror coverage for SingleCellMemMapDataset class
  python test_anndata_api_coverage.py \
    --mode mirror --class-name SingleCellMemMapDataset \
    ../../../src/bionemo/scdl/io/single_cell_memmap_dataset.py

  # Mirror coverage for all classes in a directory (per-class reports)
  python test_anndata_api_coverage.py --mode mirror ../../../src/bionemo/scdl/io/

  # Usage coverage (legacy behavior)
  python test_anndata_api_coverage.py --mode usage -v \
    ../../../src/bionemo/scdl/io/single_cell_memmap_dataset.py
"""

import ast
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Union, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import json


@dataclass
class APIUsage:
    """Represents usage of an API element."""
    name: str
    category: str
    location: str
    line_number: int


class AnnDataAPIRegistry:
    """Registry of all known AnnData API elements."""
    
    def __init__(self):
        self.api_elements = {
            # Core AnnData class attributes
            'anndata_attributes': {
                'T', 'X', 'filename', 'is_view', 'isbacked', 'layers',
                'n_obs', 'n_vars', 'obs', 'obs_names', 'obsm', 'obsp',
                'raw', 'shape', 'uns', 'var', 'var_names', 'varm', 'varp'
            },
            
            # Core AnnData class methods
            'anndata_methods': {
                'chunk_X', 'chunked_X', 'concatenate', 'copy', 'obs_keys',
                'obs_names_make_unique', 'obs_vector', 'obsm_keys',
                'rename_categories', 'strings_to_categoricals', 'to_df',
                'to_memory', 'transpose', 'uns_keys', 'var_keys',
                'var_names_make_unique', 'var_vector', 'varm_keys',
                'write', 'write_csvs', 'write_h5ad', 'write_loom', 'write_zarr'
            },
            
            # Top-level functions
            'anndata_functions': {
                'concat', 'read', 'read_h5ad', 'read_csv', 'read_excel',
                'read_hdf', 'read_loom', 'read_mtx', 'read_text', 'read_umi_tools',
                'read_zarr', 'write_elem', 'read_elem'
            },
            
            # Concatenation function parameters
            'concat_parameters': {
                'join', 'merge', 'uns_merge', 'label', 'keys', 'index_unique', 'pairwise'
            },
            
            # File format encoding types
            'encoding_types': {
                'anndata', 'array', 'csr_matrix', 'csc_matrix', 'dataframe',
                'dict', 'categorical', 'string', 'string-array', 'numeric-scalar',
                'nullable-integer', 'nullable-boolean', 'awkward-array'
            },
            
            # AnnData constructor and class
            'anndata_class': {'AnnData'},
            
            # Common import aliases
            'import_aliases': {'ad', 'anndata'},
        }
        # Categories applicable to mirror coverage by default
        self.mirror_categories_default = {
            'anndata_attributes',
            'anndata_methods',
            # Intentionally exclude: 'anndata_functions', 'encoding_types',
            # 'anndata_class', and 'import_aliases' from default mirror scoring
        }
    
    def get_all_elements(self) -> Set[str]:
        """Get all API elements across all categories."""
        all_elements = set()
        for category_elements in self.api_elements.values():
            all_elements.update(category_elements)
        return all_elements
    
    def categorize_element(self, element: str) -> str:
        """Return the category of an API element."""
        for category, elements in self.api_elements.items():
            if element in elements:
                return category
        return 'unknown'

    def elements_for_categories(self, categories: Set[str]) -> Dict[str, Set[str]]:
        return {c: set(self.api_elements[c]) for c in categories if c in self.api_elements}


class PythonASTAnalyzer(ast.NodeVisitor):
    """AST visitor to analyze Python code for AnnData API usage."""
    
    def __init__(self, file_path: str, api_registry: AnnDataAPIRegistry):
        self.file_path = file_path
        self.api_registry = api_registry
        self.api_usage: List[APIUsage] = []
        self.imports: Dict[str, str] = {}  # alias -> module
        self.anndata_aliases: Set[str] = set()
        self.anndata_instance_vars: Set[str] = set()  # variables known to be AnnData instances
        
    def visit_Import(self, node: ast.Import):
        """Track import statements."""
        for alias in node.names:
            module_name = alias.name
            import_alias = alias.asname or alias.name
            self.imports[import_alias] = module_name
            
            if module_name == 'anndata':
                self.anndata_aliases.add(import_alias)
        
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Track from...import statements."""
        if node.module == 'anndata':
            for alias in node.names:
                name = alias.name
                import_alias = alias.asname or name
                self.imports[import_alias] = f"anndata.{name}"
                
                # Track if importing AnnData class or functions directly
                if name in self.api_registry.api_elements['anndata_class']:
                    self.anndata_aliases.add(import_alias)
                elif name in self.api_registry.api_elements['anndata_functions']:
                    self._record_usage(import_alias, 'anndata_functions', node.lineno)
        
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign):
        """Track assignments creating AnnData instances via read_* or AnnData()."""
        try:
            if isinstance(node.value, ast.Call):
                # Detect ad.read_h5ad, anndata.read_*, or AnnData constructor
                func = node.value.func
                is_anndata_ctor_or_reader = False
                if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                    base = func.value.id
                    attr = func.attr
                    if base in self.anndata_aliases and (
                        attr in self.api_registry.api_elements['anndata_functions']
                        or attr in self.api_registry.api_elements['anndata_class']
                    ):
                        is_anndata_ctor_or_reader = True
                elif isinstance(func, ast.Name):
                    # from anndata import AnnData; AnnData(...)
                    fn_name = func.id
                    if fn_name in self.imports and self.imports[fn_name].startswith('anndata.'):
                        actual = self.imports[fn_name].split('.')[-1]
                        if actual in self.api_registry.api_elements['anndata_functions'] or actual in self.api_registry.api_elements['anndata_class']:
                            is_anndata_ctor_or_reader = True

                if is_anndata_ctor_or_reader:
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            self.anndata_instance_vars.add(target.id)
                        elif isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == 'self':
                            # self.adata = anndata.read_h5ad(...)
                            self.anndata_instance_vars.add(target.attr)
        finally:
            self.generic_visit(node)
    
    def visit_Call(self, node: ast.Call):
        """Track function/method calls."""
        # Handle direct function calls (e.g., ad.concat, anndata.AnnData)
        if isinstance(node.func, ast.Attribute):
            self._handle_attribute_call(node)
        elif isinstance(node.func, ast.Name):
            self._handle_name_call(node)
        
        self.generic_visit(node)
    
    def visit_Attribute(self, node: ast.Attribute):
        """Track attribute access."""
        if isinstance(node.value, ast.Name):
            # Check if this is accessing an AnnData attribute/method
            obj_name = node.value.id
            attr_name = node.attr
            
            # Check if object was created from AnnData or is an alias (ad, anndata)
            if obj_name in self.anndata_instance_vars or obj_name in self.anndata_aliases:
                category = self.api_registry.categorize_element(attr_name)
                if category != 'unknown':
                    self._record_usage(attr_name, category, node.lineno)
        
        self.generic_visit(node)
    
    def _handle_attribute_call(self, node: ast.Call):
        """Handle calls like ad.concat() or adata.write()."""
        if isinstance(node.func.value, ast.Name):
            obj_name = node.func.value.id
            method_name = node.func.attr
            
            if obj_name in self.anndata_aliases:
                # This is a call like ad.concat() or ad.AnnData()
                category = self.api_registry.categorize_element(method_name)
                if category != 'unknown':
                    self._record_usage(method_name, category, node.lineno)
            elif obj_name in self.anndata_instance_vars:
                # This is a method call on an AnnData object
                category = self.api_registry.categorize_element(method_name)
                if category != 'unknown':
                    self._record_usage(method_name, category, node.lineno)
    
    def _handle_name_call(self, node: ast.Call):
        """Handle direct calls like AnnData() or concat()."""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            
            # Check if this is a direct import (e.g., from anndata import AnnData)
            if func_name in self.imports:
                module = self.imports[func_name]
                if module.startswith('anndata.'):
                    actual_name = module.split('.')[-1]
                    category = self.api_registry.categorize_element(actual_name)
                    if category != 'unknown':
                        self._record_usage(actual_name, category, node.lineno)
    
    def _record_usage(self, element: str, category: str, line_number: int):
        """Record usage of an API element."""
        usage = APIUsage(
            name=element,
            category=category,
            location=self.file_path,
            line_number=line_number
        )
        self.api_usage.append(usage)
    
    def get_usage_summary(self) -> Dict[str, List[APIUsage]]:
        """Get summary of API usage by category."""
        summary = defaultdict(list)
        for usage in self.api_usage:
            summary[usage.category].append(usage)
        return dict(summary)


class APIReportGenerator:
    """Generates reports about API coverage."""
    
    def __init__(self, api_registry: AnnDataAPIRegistry):
        self.api_registry = api_registry
    
    def generate_coverage_report(self, used_by_category: Dict[str, Set[str]], include_categories: Optional[Set[str]] = None) -> Dict:
        """Generate a comprehensive coverage report from a category->used set mapping.

        include_categories: if provided, limit coverage to these categories (mirror mode default)
        """
        if include_categories is None:
            categories = set(self.api_registry.api_elements.keys())
        else:
            categories = include_categories

        coverage_by_category: Dict[str, Dict[str, Union[List[str], float]]] = {}
        total_elements = 0
        total_used = 0

        for category in categories:
            elements = set(self.api_registry.api_elements.get(category, set()))
            used = used_by_category.get(category, set()) if used_by_category else set()
            used_in_category = used.intersection(elements)
            total_elements += len(elements)
            total_used += len(used_in_category)
            coverage_by_category[category] = {
                'used': sorted(list(used_in_category)),
                'unused': sorted(list(elements - used_in_category)),
                'coverage_percent': (len(used_in_category) / len(elements) * 100) if elements else 0.0,
            }

        overall_percent = (total_used / total_elements * 100) if total_elements else 0.0
        return {
            'overall': {
                'total_elements': total_elements,
                'used_elements': total_used,
                'coverage_percent': overall_percent,
            },
            'by_category': coverage_by_category,
        }
    
    def print_report(self, report: Dict, verbose: bool = False, title: str = "AnnData API Coverage Report"):
        """Print a human-readable coverage report."""
        overall = report['overall']
        
        print("=" * 60)
        print(title)
        print("=" * 60)
        print(f"Overall Coverage: {overall['coverage_percent']:.1f}% "
              f"({overall['used_elements']}/{overall['total_elements']} elements)")
        print()
        
        print("Coverage by Category:")
        print("-" * 40)
        for category, data in report['by_category'].items():
            print(f"{category.replace('_', ' ').title()}: "
                  f"{data['coverage_percent']:.1f}% "
                  f"({len(data['used'])}/{len(data['used']) + len(data['unused'])})")
            
            if verbose and data['used']:
                print(f"  Used: {', '.join(sorted(data['used']))}")
            if verbose and data['unused']:
                print(f"  Unused: {', '.join(sorted(data['unused']))}")
            print()


class MirrorAnalyzer(ast.NodeVisitor):
    """Analyze a Python file to find classes and determine API surface mirroring."""

    def __init__(self, file_path: str, api_registry: AnnDataAPIRegistry, target_class_names: Optional[Set[str]] = None):
        self.file_path = file_path
        self.api_registry = api_registry
        self.target_class_names = target_class_names  # if None, analyze all classes
        self.class_to_methods: Dict[str, Set[str]] = {}
        self.class_to_attributes: Dict[str, Set[str]] = {}
        self._current_class: Optional[str] = None

    def visit_ClassDef(self, node: ast.ClassDef):
        class_name = node.name
        if self.target_class_names is not None and class_name not in self.target_class_names:
            return  # skip non-target classes

        self._current_class = class_name
        self.class_to_methods.setdefault(class_name, set())
        self.class_to_attributes.setdefault(class_name, set())

        # Walk class body
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method_name = item.name
                # @property turns a method into an attribute for API surface
                if any(isinstance(dec, ast.Name) and dec.id == 'property' for dec in item.decorator_list):
                    self.class_to_attributes[class_name].add(method_name)
                else:
                    self.class_to_methods[class_name].add(method_name)

                # Collect attributes assigned to self in __init__ as attributes
                if method_name == '__init__':
                    for stmt in ast.walk(item):
                        if isinstance(stmt, ast.Assign):
                            for target in stmt.targets:
                                if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == 'self':
                                    self.class_to_attributes[class_name].add(target.attr)

        # Continue visiting nested defs if any
        self.generic_visit(node)

    def get_used_by_category_for_class(self, class_name: str) -> Dict[str, Set[str]]:
        """Map AnnData categories to mirrored names for a given class."""
        methods = self.class_to_methods.get(class_name, set())
        attrs = self.class_to_attributes.get(class_name, set())
        used: Dict[str, Set[str]] = {
            'anndata_methods': set(name for name in methods if name in self.api_registry.api_elements['anndata_methods']),
            'anndata_attributes': set(name for name in attrs if name in self.api_registry.api_elements['anndata_attributes']),
        }
        return used


def analyze_file_usage(file_path: Path, api_registry: AnnDataAPIRegistry) -> List[APIUsage]:
    """Analyze a single Python file for AnnData API usage."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content, filename=str(file_path))
        analyzer = PythonASTAnalyzer(str(file_path), api_registry)
        analyzer.visit(tree)
        
        return analyzer.api_usage
        
    except (SyntaxError, UnicodeDecodeError) as e:
        print(f"Warning: Could not parse {file_path}: {e}", file=sys.stderr)
        return []


def analyze_directory_usage(directory: Path, api_registry: AnnDataAPIRegistry) -> List[APIUsage]:
    """Recursively analyze all Python files in a directory."""
    all_usage = []
    
    for py_file in directory.rglob("*.py"):
        usage = analyze_file_usage(py_file, api_registry)
        all_usage.extend(usage)
    
    return all_usage


def analyze_file_mirror(file_path: Path, api_registry: AnnDataAPIRegistry, class_names: Optional[List[str]] = None) -> Dict[str, Dict[str, Set[str]]]:
    """Analyze a single Python file for AnnData API mirroring.

    Returns a mapping class_name -> used_by_category
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        tree = ast.parse(content, filename=str(file_path))
        targets = set(class_names) if class_names else None
        analyzer = MirrorAnalyzer(str(file_path), api_registry, targets)
        analyzer.visit(tree)
        result: Dict[str, Dict[str, Set[str]]] = {}
        for class_name in analyzer.class_to_methods.keys() | analyzer.class_to_attributes.keys():
            result[class_name] = analyzer.get_used_by_category_for_class(class_name)
        return result
    except (SyntaxError, UnicodeDecodeError) as e:
        print(f"Warning: Could not parse {file_path}: {e}", file=sys.stderr)
        return {}


def analyze_directory_mirror(directory: Path, api_registry: AnnDataAPIRegistry, class_names: Optional[List[str]] = None) -> Dict[str, Dict[str, Set[str]]]:
    """Recursively analyze all Python files in a directory for mirror coverage.

    Returns mapping class_name -> used_by_category (aggregated across files if duplicate class names occur, last wins)
    """
    all_results: Dict[str, Dict[str, Set[str]]] = {}
    for py_file in directory.rglob("*.py"):
        file_results = analyze_file_mirror(py_file, api_registry, class_names)
        all_results.update(file_results)
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Python code for AnnData API coverage: usage (calls) or mirror (API parity)"
    )
    parser.add_argument(
        "path",
        help="Path to Python file or directory to analyze"
    )
    parser.add_argument(
        "--mode",
        choices=["usage", "mirror"],
        default="mirror",
        help="Analysis mode: 'usage' (detect calls to AnnData API) or 'mirror' (detect mirrored AnnData API on classes)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed usage information"
    )
    parser.add_argument(
        "--class-name",
        action="append",
        help="Class name to analyze for mirror coverage (can be provided multiple times). If omitted in mirror mode, analyze all classes found."
    )
    parser.add_argument(
        "-o", "--output",
        help="Output report to JSON file"
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.0,
        help="Minimum coverage percentage (exit with error if below)"
    )
    
    args = parser.parse_args()
    
    path = Path(args.path)
    if not path.exists():
        print(f"Error: Path {path} does not exist", file=sys.stderr)
        sys.exit(1)
    
    api_registry = AnnDataAPIRegistry()
    print(f"Analyzing {path}...")

    report_generator = APIReportGenerator(api_registry)

    if args.mode == 'usage':
        # Usage mode: legacy behavior
        if path.is_file():
            all_usage = analyze_file_usage(path, api_registry)
        else:
            all_usage = analyze_directory_usage(path, api_registry)

        # Build used_by_category from APIUsage list
        used_by_category: Dict[str, Set[str]] = {}
        for usage in all_usage:
            used_by_category.setdefault(usage.category, set()).add(usage.name)
        report = report_generator.generate_coverage_report(used_by_category)
        report_generator.print_report(report, verbose=args.verbose, title="AnnData API Coverage Report (usage)")
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\nReport saved to {args.output}")
        coverage = report['overall']['coverage_percent']
        if coverage < args.min_coverage:
            print(f"\nError: Coverage {coverage:.1f}% is below minimum {args.min_coverage}%", file=sys.stderr)
            sys.exit(1)
        return

    # Mirror mode
    include_categories = api_registry.mirror_categories_default
    class_names = args.class_name

    # Analyze mirroring
    if path.is_file():
        class_to_used = analyze_file_mirror(path, api_registry, class_names)
    else:
        class_to_used = analyze_directory_mirror(path, api_registry, class_names)

    if not class_to_used:
        print("No target classes found for mirror analysis.")
        sys.exit(1)

    # Print per-class reports and compute worst coverage vs min threshold
    worst_coverage = 100.0
    for cls, used_by_category in class_to_used.items():
        report = report_generator.generate_coverage_report(used_by_category, include_categories)
        report_generator.print_report(report, verbose=args.verbose, title=f"AnnData API Mirror Coverage Report: class {cls}")
        if args.output:
            # Write per-class report into separate JSON files or a single dict
            out_path = Path(args.output)
            if out_path.suffix:
                # If output is a file path, write a dict combining classes
                combined = {}
                if out_path.exists():
                    try:
                        with open(out_path, 'r') as rf:
                            combined = json.load(rf)
                    except Exception:
                        combined = {}
                combined[cls] = report
                with open(out_path, 'w') as wf:
                    json.dump(combined, wf, indent=2)
            else:
                # Treat as directory
                out_path.mkdir(parents=True, exist_ok=True)
                with open(out_path / f"{cls}_mirror_report.json", 'w') as wf:
                    json.dump(report, wf, indent=2)
        worst_coverage = min(worst_coverage, report['overall']['coverage_percent'])

    if worst_coverage < args.min_coverage:
        print(f"\nError: Coverage {worst_coverage:.1f}% is below minimum {args.min_coverage}%", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()