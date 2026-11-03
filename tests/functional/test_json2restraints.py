import re
from pathlib import Path
import pytest
from tutorial.json2restraints import main as json2restraints_main


def _assert_exte_dist_syntax(text_output: str) -> None:
    pattern = (
        r"^exte dist first chain [A-Za-z0-9]+ resi \d+ inse \. atom "
        r"[A-Za-z0-9]+(?: altecode [A-Za-z0-9])? "
        r"second chain [A-Za-z0-9]+ resi \d+ inse \. atom "
        r"[A-Za-z0-9]+(?: altecode [A-Za-z0-9])? "
        r"value [0-9]+(?:\.[0-9]+)? sigma [0-9]+(?:\.[0-9]+)? "
        r"(?:symm y )?"
        r"(?:type [0-9]+)?"
        r"(?: # .*?)?$"
    )

    dist_lines = [line.strip() for line in text_output.splitlines() if line.startswith("exte dist ")]
    assert dist_lines, "No exte dist lines were written"

    for line in dist_lines:
        assert re.match(pattern, line), f"Unexpected exte dist syntax: {line}"


def _assert_exte_angle_syntax(text_output: str) -> None:
    pattern = (
        r"^exte angle first chain [A-Za-z0-9]+ resi \d+ inse \. atom "
        r"[A-Za-z0-9]+(?: altecode [A-Za-z0-9])?(?: symm y)? "
        r"next chain [A-Za-z0-9]+ resi \d+ inse \. atom "
        r"[A-Za-z0-9]+(?: altecode [A-Za-z0-9])? "
        r"next chain [A-Za-z0-9]+ resi \d+ inse \. atom "
        r"[A-Za-z0-9]+(?: altecode [A-Za-z0-9])?(?: symm y)? "
        r"value [0-9]+(?:\.[0-9]+)? sigma [0-9]+(?:\.[0-9]+)? "
        r"(?:type [0-9]+)?"
        r"(?: # .*?)?$"
    )

    angle_lines = [line.strip() for line in text_output.splitlines() if line.startswith("exte angle ")]
    assert angle_lines, "No exte angle lines were written"

    for line in angle_lines:
        assert re.match(pattern, line), f"Unexpected exte angle syntax: {line}"


def run_servalcat_refine_geom(output_prefix: Path) -> None:
    try:
        from servalcat.refine import refine_geom as servalcat_refine_geom
    except ModuleNotFoundError:
        pytest.skip("servalcat is not available; skipping refine_geom validation")
    except Exception as exc:
        pytest.skip(f"servalcat import failed: {exc!r}")

    try:
        args = servalcat_refine_geom.parse_args([
            "--model",
            str(output_prefix.with_suffix(".mmcif")),
            "--keyword_file",
            str(output_prefix.with_suffix(".txt")),
            "--output_prefix",
            str(output_prefix) + "_refine_geom",
            "--ncycle",
            "0",
        ])
        servalcat_refine_geom.main(args)
    except Exception as exc:
        pytest.fail(f"servalcat refine_geom failed: {exc!r}")


def test_json2restraints_3kw8(tmp_path):
    project_root = Path(__file__).resolve().parents[2]
    json_path = project_root / "tests" / "data" / "results" / "3kw8_edit_alte_CU-d04.json"
    pdb_path = project_root / "tests" / "data" / "models" / "3kw8_edit_alte.pdb"
    output_prefix = tmp_path / "3kw8_mc_restraints"

    json2restraints_main(
        jsonPaths=[str(json_path)],
        stPath=str(pdb_path),
        outputPrefix=str(output_prefix),
    )

    expected_files = [
        output_prefix.with_suffix(".txt"),
        output_prefix.with_name(output_prefix.name + "_coot.txt"),
        output_prefix.with_suffix(".params"),
        output_prefix.with_suffix(".mmcif"),
        output_prefix.with_suffix(".pdb"),
    ]

    for path in expected_files:
        assert path.exists(), f"Missing generated file: {path}"

    output_restraints = output_prefix.with_suffix(".txt").read_text(encoding="utf-8")
    _assert_exte_dist_syntax(output_restraints)
    _assert_exte_angle_syntax(output_restraints)
    assert output_restraints.count("exte dist") >= 14
    assert output_restraints.count("exte angle") >= 8
    assert output_restraints.count("symm y") >= 15  #  A404 CU altecode B

    output_coot = output_prefix.with_name(output_prefix.name + "_coot.txt").read_text(encoding="utf-8")
    assert "#" not in output_coot
    assert "symm y" not in output_coot

    run_servalcat_refine_geom(output_prefix)
