import json
import re
import warnings
from pathlib import Path
import gemmi
import pytest
from tutorial.json2restraints import main as json2restraints_main

# Optional tools: the checks using them are run only if they are available
try:
    # the import order matters: servalcat.refine.refine imported first leads to a circular import
    from omegaconf import OmegaConf
    from servalcat import ext
    from servalcat.refmac import refmac_keywords

    try:
        from servalcat.refine import exte
    except ImportError:
        from servalcat.refmac import exte  # older servalcat versions
    from servalcat.refine.refine import RefineConfig, RefineParams

    SERVALCAT_AVAIL = True
    # older servalcat versions (e.g. 0.4.142 in CCP4 9) do not support external restraints in JSON
    SERVALCAT_JSON_AVAIL = "exte_files" in RefineConfig.__dataclass_fields__
except ImportError:
    SERVALCAT_AVAIL = SERVALCAT_JSON_AVAIL = False
try:
    import iotbx.pdb
    import iotbx.phil
    from mmtbx.monomer_library.pdb_interpretation import geometry_restraints_edits_str

    IOTBX_AVAIL = True
except ImportError:
    IOTBX_AVAIL = False


EXTE_DIST_PATTERN = (
    r"^exte dist first chain [A-Za-z0-9]+ resi \d+ inse [A-Za-z.] atom "
    r"[A-Za-z0-9]+(?: altecode [A-Za-z0-9])? "
    r"second chain [A-Za-z0-9]+ resi \d+ inse [A-Za-z.] atom "
    r"[A-Za-z0-9]+(?: altecode [A-Za-z0-9])? "
    r"value [0-9]+(?:\.[0-9]+)? sigma [0-9]+(?:\.[0-9]+)? "
    r"(?:symm y )?"
    r"(?:type [0-9]+)?"
    r"(?: # .*?)?$"
)
EXTE_ANGLE_PATTERN = (
    r"^exte angle first chain [A-Za-z0-9]+ resi \d+ inse [A-Za-z.] atom "
    r"[A-Za-z0-9]+(?: altecode [A-Za-z0-9])?(?: symm y)? "
    r"next chain [A-Za-z0-9]+ resi \d+ inse [A-Za-z.] atom "
    r"[A-Za-z0-9]+(?: altecode [A-Za-z0-9])? "
    r"next chain [A-Za-z0-9]+ resi \d+ inse [A-Za-z.] atom "
    r"[A-Za-z0-9]+(?: altecode [A-Za-z0-9])?(?: symm y)? "
    r"value [0-9]+(?:\.[0-9]+)? sigma [0-9]+(?:\.[0-9]+)? "
    r"(?:type [0-9]+)?"
    r"(?: # .*?)?$"
)


def flatten(obj, path=()) -> set:
    """Flatten nested dicts/lists into a set of (path, value) pairs, list lengths included.
    Restraint A is contained in restraint B (B may have extra keys) if flatten(A) <= flatten(B).
    This is used to check that every restraint parsed by servalcat from the .txt keyword file is in the JSON file.
    """
    if isinstance(obj, dict):
        return set().union(*(flatten(v, path + (k,)) for k, v in obj.items()))
    if isinstance(obj, list):
        return {(path, len(obj))}.union(*(flatten(v, path + (i,)) for i, v in enumerate(obj)))
    return {(path, obj)}


@pytest.mark.parametrize("case", ["3kw8", "1e8a"])
def test_json2restraints(case, tmp_path):
    data = Path(__file__).resolve().parents[2] / "tests" / "data"
    if case == "3kw8":
        # Cu sites; A404 CU has altloc B and insertion code C, some ligands are symmetry mates
        json_path = data / "results" / "3kw8_edit_alte_CU-d04.json"
        model_path = data / "models" / "3kw8_edit_alte.pdb.gz"
    elif case == "1e8a":
        # Ca sites; for A1090 CA, MetalCoord grouped bidentate ligands into centroids
        json_path = data / "models" / "1E8A_CA.json"
        model_path = data / "models" / "1E8A.cif.gz"
    output_prefix = tmp_path / f"{case}_mc_restraints"

    json2restraints_main(
        jsonPaths=[str(json_path)],
        stPath=str(model_path),
        outputPrefix=str(output_prefix),
    )

    txt_path = output_prefix.with_suffix(".txt")
    json_path = output_prefix.with_suffix(".json")
    yaml_path = output_prefix.with_suffix(".yaml")
    coot_path = output_prefix.with_name(output_prefix.name + "_coot.txt")
    buster_path = output_prefix.with_name(output_prefix.name + "_buster.txt")
    params_path = output_prefix.with_suffix(".params")
    mmcif_path = output_prefix.with_suffix(".mmcif")
    pdb_path = output_prefix.with_suffix(".pdb")
    for path in [txt_path, json_path, yaml_path, coot_path, buster_path, params_path, mmcif_path, pdb_path]:
        assert path.exists(), f"Missing generated file: {path}"

    output_txt = txt_path.read_text(encoding="utf-8")
    output_json = json.loads(json_path.read_text(encoding="utf-8"))
    output_yaml = yaml_path.read_text(encoding="utf-8")
    output_coot = coot_path.read_text(encoding="utf-8")
    output_buster = buster_path.read_text(encoding="utf-8")

    # --- Servalcat/Refmac keywords (.txt) ---
    dist_lines = [line.strip() for line in output_txt.splitlines() if line.startswith("exte dist ")]
    angle_lines = [line.strip() for line in output_txt.splitlines() if line.startswith("exte angle ")]
    assert dist_lines and angle_lines
    for line in dist_lines:
        assert re.match(EXTE_DIST_PATTERN, line), f"Unexpected exte dist syntax: {line}"
    for line in angle_lines:
        assert re.match(EXTE_ANGLE_PATTERN, line), f"Unexpected exte angle syntax: {line}"

    # --- Coot: no comments, no symmetry ---
    assert "#" not in output_coot
    assert "symm y" not in output_coot

    # --- Servalcat JSON and YAML config ---
    dist_json = [r for r in output_json if r["rest_type"] == "dist"]
    angl_json = [r for r in output_json if r["rest_type"] == "angl"]
    cdist_json = [r for r in output_json if r["rest_type"] == "cdist"]
    cangl_json = [r for r in output_json if r["rest_type"] == "cangl"]
    assert len(dist_json) == len(dist_lines)
    assert len(angl_json) == len(angle_lines)
    assert len(output_json) == len(dist_json) + len(angl_json) + len(cdist_json) + len(cangl_json)
    for r in dist_json + cdist_json:
        assert len(r["restr"]["specs"]) == 2
    for r in angl_json + cangl_json:
        assert len(r["restr"]["specs"]) == 3
    assert "exte_files" in output_yaml
    assert str(json_path) in output_yaml

    # --- data set specific checks ---
    if case == "3kw8":
        assert len(dist_lines) >= 14
        assert len(angle_lines) >= 8
        assert output_txt.count("symm y") >= 15
        assert not cdist_json and not cangl_json

        # A404C CU altecode B: insertion code and altloc must be propagated to all formats
        assert "resi 404 inse C atom CU altecode B" in output_txt
        assert "A|404C:CU.B" in output_buster
        assert "resid 404C and name CU and altloc B" in params_path.read_text(encoding="utf-8")
        specs_404 = [spec for r in output_json for specs in r["restr"]["specs"] for spec in specs if spec["resi"] == 404]
        assert specs_404
        for spec in specs_404:
            assert spec.get("ins") == "C" and spec.get("alt") == "B", spec

    if case == "1e8a":
        # A1090 CA: octahedral with 2 groups collapsed (ASP 65 OD1/OD2 and GLU 72 OE1/OE2)
        # centroid restraints are only in the JSON file
        assert "cdist" not in output_txt and "cangl" not in output_txt
        assert len(cdist_json) == 2
        assert len(cangl_json) == 9
        metal = [{"chain": "A", "resi": 1090, "names": ["CA"]}]
        for r in cdist_json:
            assert r["restr"]["specs"][0] == metal
        for r in cangl_json:
            assert r["restr"]["specs"][1] == metal
        centroid_members = {(spec["resi"], spec["names"][0]) for r in cdist_json for spec in r["restr"]["specs"][1]}
        assert centroid_members == {(65, "OD1"), (65, "OD2"), (72, "OE1"), (72, "OE2")}

    # --- Servalcat: load the restraints for the output structure, no refinement run ---
    # read_external_restraints() raises RuntimeError if a restraint refers to an atom not in the structure
    if SERVALCAT_AVAIL:
        st = gemmi.read_structure(str(mmcif_path))

        # restraints from the .txt keyword file (as servalcat --keyword_file)
        geom = ext.Geometry(st, RefineParams(st, refine_xyz=True), gemmi.EnerLib())
        if SERVALCAT_JSON_AVAIL:
            exte_blocks = refmac_keywords.RefmacKeywords(keyword_files=[[str(txt_path)]]).params["exte_blocks"]
            exte.read_external_restraints(exte_blocks, st, geom)
        else:
            # older servalcat: a list of restraints instead of blocks from individual files
            exte_list = refmac_keywords.parse_keywords(txt_path.read_text(encoding="utf-8").splitlines())["exte"]
            exte.read_external_restraints(exte_list, st, geom)
        assert len(geom.bonds) == len(dist_lines)
        assert len(geom.angles) == len(angle_lines)

        if SERVALCAT_JSON_AVAIL:
            # restraints from the .json file given in the .yaml config (as servalcat --config)
            config = OmegaConf.merge(OmegaConf.create({"refine": RefineConfig()}), OmegaConf.load(yaml_path))
            exte_blocks = {f: json.loads(Path(f).read_text(encoding="utf-8")) for f in config.refine.exte_files}
            geom = ext.Geometry(st, RefineParams(st, refine_xyz=True), gemmi.EnerLib())
            exte.read_external_restraints(exte_blocks, st, geom)
            assert len(geom.bonds) == len(dist_json)
            assert len(geom.angles) == len(angl_json)
            assert len(geom.cbonds) == len(cdist_json)
            assert len(geom.cangles) == len(cangl_json)

            # every restraint parsed by servalcat from the .txt keyword file must be in the JSON file
            # (the JSON file may contain more: extra keys or extra restraints like cdist and cangl)
            json_flat = [flatten(r) for r in output_json]
            for line in dist_lines + angle_lines:
                r = refmac_keywords.read_exte(line.split("#")[0].split())
                r.pop("defaults", None)
                assert any(flatten(r) <= f for f in json_flat), f"Not found in the JSON file: {line}"
        else:
            warnings.warn("servalcat is too old for JSON restraints: JSON checks not run")
    else:
        warnings.warn("servalcat not available: Servalcat checks not run")

    # --- Phenix (.params): syntax and atom selections, no refinement run ---
    if IOTBX_AVAIL:
        # Phenix definition of the parameters (names, types, allowed values) for the .params file
        master = iotbx.phil.parse(f"refinement.geometry_restraints.edits {{\n{geometry_restraints_edits_str}\n}}")
        params = iotbx.phil.parse(file_name=str(params_path))
        working, unused = master.fetch(sources=[params], track_unused_definitions=True)
        assert not unused, f"Unknown Phenix parameters: {[str(u) for u in unused]}"
        edits = working.extract().refinement.geometry_restraints.edits
        assert edits.bond and edits.angle

        # every atom selection must select exactly one atom
        cache = iotbx.pdb.input(file_name=str(pdb_path)).construct_hierarchy().atom_selection_cache()
        selections = [s for r in edits.bond for s in (r.atom_selection_1, r.atom_selection_2)]
        selections += [s for r in edits.angle for s in (r.atom_selection_1, r.atom_selection_2, r.atom_selection_3)]
        for selection in selections:
            n = cache.selection(selection).count(True)
            assert n == 1, f"Phenix selection matches {n} atoms: {selection}"
    else:
        warnings.warn("iotbx not available: Phenix checks not run")
