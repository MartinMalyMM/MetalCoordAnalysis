import json
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


def run_servalcat_refine_geom(output_prefix: Path, use_config: bool = False) -> None:
    """Run servalcat refine_geom with the generated restraints.

    use_config=False: restraints from the .txt keyword file (--keyword_file)
    use_config=True:  restraints from the .json file via the .yaml config (--config)
    """
    try:
        from servalcat.refine import refine_geom as servalcat_refine_geom
    except ModuleNotFoundError:
        pytest.skip("servalcat is not available; skipping refine_geom validation")
    except Exception as exc:
        pytest.skip(f"servalcat import failed: {exc!r}")

    if use_config:
        restraints_args = ["--config", str(output_prefix.with_suffix(".yaml"))]
        run_suffix = "_refine_geom_config"
    else:
        restraints_args = ["--keyword_file", str(output_prefix.with_suffix(".txt"))]
        run_suffix = "_refine_geom"

    try:
        args = servalcat_refine_geom.parse_args([
            "--model",
            str(output_prefix.with_suffix(".mmcif")),
            *restraints_args,
            "--output_prefix",
            str(output_prefix) + run_suffix,
            "--ncycle",
            "0",
        ])
        servalcat_refine_geom.main(args)
    except Exception as exc:
        pytest.fail(f"servalcat refine_geom ({' '.join(restraints_args)}) failed: {exc!r}")


def _flatten(obj, path=()) -> set:
    """Flatten nested dicts/lists into a set of (path, value) pairs.
    For lists, (path, length) is included too, so lists must match in length."""
    if isinstance(obj, dict):
        return set().union(*(_flatten(v, path + (k,)) for k, v in obj.items()))
    if isinstance(obj, list):
        return {(path, len(obj))}.union(*(_flatten(v, path + (i,)) for i, v in enumerate(obj)))
    return {(path, obj)}


def _assert_json_matches_txt(output_prefix: Path) -> None:
    """Every restraint servalcat parses from the .txt keyword file must be present in the JSON file.
    The JSON file may contain more information (extra keys or extra restraints) than the .txt file."""
    try:
        from servalcat.refmac import refmac_keywords
    except ModuleNotFoundError:
        pytest.skip("servalcat is not available; skipping JSON/keyword comparison")
    except Exception as exc:
        pytest.skip(f"servalcat import failed: {exc!r}")

    restraints_json = json.loads(output_prefix.with_suffix(".json").read_text(encoding="utf-8"))
    json_flat = [_flatten(r) for r in restraints_json]
    for line in output_prefix.with_suffix(".txt").read_text(encoding="utf-8").splitlines():
        if not line.startswith("exte "):
            continue
        r = refmac_keywords.read_exte(line.split("#")[0].split())
        r.pop("defaults", None)
        assert any(_flatten(r) <= f for f in json_flat), f"Not found in the JSON file: {line}"


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
        output_prefix.with_suffix(".json"),
        output_prefix.with_suffix(".yaml"),
        output_prefix.with_name(output_prefix.name + "_coot.txt"),
        output_prefix.with_name(output_prefix.name + "_buster.txt"),
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

    restraints_json = json.loads(output_prefix.with_suffix(".json").read_text(encoding="utf-8"))
    dist_json = [r for r in restraints_json if r["rest_type"] == "dist"]
    angl_json = [r for r in restraints_json if r["rest_type"] == "angl"]
    assert len(dist_json) >= output_restraints.count("exte dist")
    assert len(angl_json) >= output_restraints.count("exte angle")
    for r in dist_json:
        assert len(r["restr"]["specs"]) == 2
    for r in angl_json:
        assert len(r["restr"]["specs"]) == 3
    assert any(
        spec.get("alt") == "B"
        for r in restraints_json
        for specs in r["restr"]["specs"]
        for spec in specs
    ), "No restraint with altloc B in the JSON file"

    output_yaml = output_prefix.with_suffix(".yaml").read_text(encoding="utf-8")
    assert "exte_files" in output_yaml
    assert str(output_prefix.with_suffix(".json")) in output_yaml

    _assert_json_matches_txt(output_prefix)
    run_servalcat_refine_geom(output_prefix)
    run_servalcat_refine_geom(output_prefix, use_config=True)
