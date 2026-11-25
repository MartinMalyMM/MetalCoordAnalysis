# servalcat, iotbx and mmtbx have no type information (optional tools, see below)
# mypy: disable-error-code="import-untyped,import-not-found"
from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any
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

# Buster (refine -restraints): add bond|angle <target> <sigma> <atom-1> ... <atom-N>,
# atoms as <chain>|<resnum>[<icode>]:<atmnam>[.<altConf>], angles in radians
BUSTER_ATOM = r"[A-Za-z0-9]+\|-?\d+[A-Za-z]?:[A-Za-z0-9']+(?:\.[A-Za-z0-9])?"
BUSTER_BOND_PATTERN = rf"^add bond [0-9]+(?:\.[0-9]+)? [0-9]+(?:\.[0-9]+)? {BUSTER_ATOM} {BUSTER_ATOM}$"
BUSTER_ANGLE_PATTERN = rf"^add angle [0-9]\.[0-9]{{6}} [0-9]\.[0-9]{{6}} {BUSTER_ATOM} {BUSTER_ATOM} {BUSTER_ATOM}$"

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "tests" / "data"
TUTORIAL = REPO / "tutorial"


def flatten(obj: Any, path: tuple[str | int, ...] = ()) -> set[tuple[tuple[str | int, ...], Any]]:
    """Flatten nested dicts/lists into a set of (path, value) pairs, list lengths included.
    Restraint A is contained in restraint B (B may have extra keys) if flatten(A) <= flatten(B).
    This is used to check that every restraint parsed by servalcat from the .txt keyword file is in the JSON file.
    """
    if isinstance(obj, dict):
        return set().union(*(flatten(v, path + (k,)) for k, v in obj.items()))
    if isinstance(obj, list):
        return {(path, len(obj))}.union(*(flatten(v, path + (i,)) for i, v in enumerate(obj)))
    return {(path, obj)}


def read_txt_restraints(st: gemmi.Structure, txt_path: Path) -> ext.Geometry:
    """Load the restraints from the .txt keyword file (as servalcat --keyword_file), no refinement run.
    read_external_restraints() raises RuntimeError if a restraint refers to an atom not in the structure.
    """
    geom = ext.Geometry(st, RefineParams(st, refine_xyz=True), gemmi.EnerLib())
    if SERVALCAT_JSON_AVAIL:
        exte_blocks = refmac_keywords.RefmacKeywords(keyword_files=[[str(txt_path)]]).params["exte_blocks"]
        exte.read_external_restraints(exte_blocks, st, geom)
    else:
        # older servalcat: a list of restraints instead of blocks from individual files
        exte_list = refmac_keywords.parse_keywords(txt_path.read_text(encoding="utf-8").splitlines())["exte"]
        exte.read_external_restraints(exte_list, st, geom)
    return geom


@pytest.mark.parametrize("case", ["3kw8", "1e8a"])
def test_json2restraints(case: str, tmp_path: Path) -> None:
    if case == "3kw8":
        # Cu sites; A404 CU has altloc B and insertion code C, some ligands are symmetry mates
        json_path = DATA / "results" / "3kw8_edit_alte_CU-d04.json"
        model_path = DATA / "models" / "3kw8_edit_alte.pdb.gz"
    elif case == "1e8a":
        # Ca sites; for A1090 CA, MetalCoord grouped bidentate ligands into centroids
        json_path = DATA / "models" / "1E8A_CA.json"
        model_path = DATA / "models" / "1E8A.cif.gz"
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
    pdb_buster_path = output_prefix.with_name(output_prefix.name + "_buster.pdb")
    for path in [txt_path, json_path, yaml_path, coot_path, buster_path, params_path, mmcif_path, pdb_path, pdb_buster_path]:
        assert path.exists(), f"Missing generated file: {path}"

    # --- PDB file for Buster: the same links as in the .pdb file except metal-water links ---
    def link_set(st: gemmi.Structure, metal_water: bool) -> set[tuple[str, ...]]:
        links = set()
        for c in st.connections:
            cras = [st[0].find_cra(p, ignore_segment=True) for p in (c.partner1, c.partner2)]
            is_metal_water = any(
                a.atom.element.is_metal and b.residue.is_water() for a, b in (cras, cras[::-1])
            )
            if is_metal_water == metal_water:
                links.add(tuple(sorted(str(cra) for cra in cras)))
        return links

    st_pdb = gemmi.read_structure(str(pdb_path))
    st_pdb_buster = gemmi.read_structure(str(pdb_buster_path))
    assert link_set(st_pdb, metal_water=True), "test case expected to contain metal-water links"
    assert not link_set(st_pdb_buster, metal_water=True)
    assert link_set(st_pdb_buster, metal_water=False) == link_set(st_pdb, metal_water=False)

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

    # --- Buster: syntax, angles in radians, no symmetry, one variant per atom pair ---
    buster_lines = output_buster.splitlines()
    buster_bonds = [line for line in buster_lines if line.startswith("add bond ")]
    buster_angles = [line for line in buster_lines if line.startswith("add angle ")]
    assert len(buster_bonds) + len(buster_angles) == len(buster_lines), "Unexpected lines in the Buster file"
    for line in buster_bonds:
        assert re.match(BUSTER_BOND_PATTERN, line), f"Unexpected Buster bond syntax: {line}"
    for line in buster_angles:
        assert re.match(BUSTER_ANGLE_PATTERN, line), f"Unexpected Buster angle syntax: {line}"
        target, sigma = (float(x) for x in line.split()[2:4])
        # 180 degrees is 3.141593 after rounding to 6 decimal places
        assert 0 < target <= math.pi + 1e-6 and 0 < sigma < math.pi, f"Buster angle not in radians: {line}"
    # no duplicates (Buster keeps only the last one): bonds A-B = B-A, angles A-M-B = B-M-A
    buster_atoms: list[Any] = [frozenset(line.split()[4:]) for line in buster_bonds]
    buster_atoms += [(line.split()[5], frozenset(line.split()[4:7:2])) for line in buster_angles]
    assert len(set(buster_atoms)) == len(buster_atoms), "More Buster restraints for the same atoms"
    # the same restraints as in the .txt file except those with symmetry mates and extra distance variants
    dist_pairs_nosymm = {line.split(" value ")[0] for line in dist_lines if "symm y" not in line}
    angle_lines_nosymm = [line for line in angle_lines if "symm y" not in line]
    assert len(buster_bonds) == len(dist_pairs_nosymm)
    assert len(buster_angles) == len(angle_lines_nosymm)

    # --- Phenix (.params): one variant per atom pair (Phenix keeps the last one), the same bonds as for Buster ---
    output_phenix = params_path.read_text(encoding="utf-8")
    phenix_bond_atoms = [
        frozenset(m) for m in re.findall(r"bond \{\s*action = \*add\s*atom_selection_1 = (.*)\n\s*atom_selection_2 = (.*)\n", output_phenix)
    ]
    assert len(phenix_bond_atoms) == output_phenix.count("bond {")
    assert len(set(phenix_bond_atoms)) == len(phenix_bond_atoms), "More Phenix bond restraints for the same atoms"
    assert len(phenix_bond_atoms) == len(buster_bonds)

    # --- Coot: no comments, no symmetry, no alt. codes ---
    assert "#" not in output_coot
    assert "symm y" not in output_coot
    assert "alte" not in output_coot

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
        # Buster: no symmetry mates (e.g. A102 NE2 coordinates A404C CU via symmetry),
        # one distance variant closest to the model (A404C CU.B - A930 O: 2.63 A, variants 1.78 and 2.2 A),
        # angles in radians
        assert "A|102:NE2" not in output_buster
        assert "add bond 2.2 0.15 A|404C:CU.B A|930:O\n" in output_buster
        assert "add bond 1.78 " not in output_buster
        assert "add bond 2.38 0.15 A|402:CU A|701:O\n" in output_buster
        assert "add bond 2.04 " not in output_buster
        assert "add angle 2.131745 0.146433 A|293:ND1 A|401:CU A|288:SG\n" in output_buster
        # Phenix: the same distance variants as for Buster
        assert "distance_ideal = 2.2\n" in output_phenix and "distance_ideal = 1.78\n" not in output_phenix
        assert "distance_ideal = 2.38\n" in output_phenix and "distance_ideal = 2.04\n" not in output_phenix
        assert "resid 404C and name CU and altloc B" in params_path.read_text(encoding="utf-8")
        specs_404 = [spec for r in output_json for specs in r["restr"]["specs"] for spec in specs if spec["resi"] == 404]
        assert specs_404
        for spec in specs_404:
            assert spec.get("ins") == "C" and spec.get("alt") == "B", spec

    if case == "1e8a":
        # The grouped-fallback class (Q in the descriptor) is preferred over the first listed class:
        # A1089, B1089, B1090: octahedral with 1 group collapsed (GLU 31 or GLU 72 OE1/OE2)
        # A1090: octahedral with 2 groups collapsed (ASP 65 OD1/OD2 and GLU 72 OE1/OE2)
        # centroid restraints are only in the JSON file
        assert "cdist" not in output_txt and "cangl" not in output_txt
        assert "class: octahedral" in output_txt
        assert "class: pentagonal-bipyramid" not in output_txt
        assert len(cdist_json) == 5
        assert len(cangl_json) == 24
        expected_members = {
            ("A", 1089): {(31, "OE1"), (31, "OE2")},
            ("A", 1090): {(65, "OD1"), (65, "OD2"), (72, "OE1"), (72, "OE2")},
            ("B", 1089): {(31, "OE1"), (31, "OE2")},
            ("B", 1090): {(72, "OE1"), (72, "OE2")},
        }
        for (chain, resi), members in expected_members.items():
            metal = [{"chain": chain, "resi": resi, "names": ["CA"]}]
            cdist_metal = [r for r in cdist_json if r["restr"]["specs"][0] == metal]
            assert cdist_metal, f"No cdist restraints for {chain}{resi}"
            centroid_members = {(spec["resi"], spec["names"][0]) for r in cdist_metal for spec in r["restr"]["specs"][1]}
            assert centroid_members == members
        metals = [[{"chain": c, "resi": r, "names": ["CA"]}] for c, r in expected_members]
        for r in cdist_json:
            assert r["restr"]["specs"][0] in metals
        for r in cangl_json:
            assert r["restr"]["specs"][1] in metals

        # links: for each centroid, every member atom is linked to the metal
        for model_path in (mmcif_path, pdb_path):
            st_out = gemmi.read_structure(str(model_path))
            links = {
                (c.partner1.chain_name, c.partner1.res_id.seqid.num, c.partner2.res_id.seqid.num, c.partner2.atom_name)
                for c in st_out.connections
                if c.type == gemmi.ConnectionType.MetalC and c.partner1.atom_name == "CA"
            }
            for (chain, resi), members in expected_members.items():
                linked_members = {(r, n) for c, m, r, n in links if (c, m) == (chain, resi) and (r, n) in members}
                assert linked_members == members, f"{model_path.name} {chain}{resi}"

    # --- Servalcat: load the restraints for the output structure, no refinement run ---
    if SERVALCAT_AVAIL:
        st = gemmi.read_structure(str(mmcif_path))
        geom = read_txt_restraints(st, txt_path)
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


# --- Metal-atom-atom-metal triangles: 8TNV example from tutorial.rst ---
# Cu2O2 cluster (CUO) in 8TNV: triangles CU1-O1-O2-CU1 and CU2-O1-O2-CU2 in chains A (402) and B (406)
CUO_RESIDUES = [("A", 402), ("B", 406)]


def test_triangles(tmp_path: Path) -> None:
    """triangles.py reproduces the tutorial files: in each O1-O2 pair, O1 is kept and moved
    to the midpoint and O2 is deleted (deterministically), the equivalents include "operator".
    """
    model_path = tmp_path / "8tnv.cif"
    model_path.write_bytes((DATA / "models" / "8tnv.cif").read_bytes())
    # run in tmp_path as the script writes output files next to the input
    subprocess.run(
        [sys.executable, str(TUTORIAL / "triangles.py"), model_path.name],
        cwd=tmp_path, check=True, capture_output=True,
    )

    equivalents_path = tmp_path / "8tnv_nopairs_equivalents.json"
    mmcif_path = tmp_path / "8tnv_nopairs.mmcif"
    for path in [equivalents_path, mmcif_path]:
        assert path.exists(), f"Missing generated file: {path}"

    # --- equivalents (.json) ---
    equivalents = json.loads(equivalents_path.read_text(encoding="utf-8"))
    assert equivalents == json.loads((TUTORIAL / equivalents_path.name).read_text(encoding="utf-8"))
    assert [(p[0]["chain"], p[0]["name"], p[1]["name"]) for p in equivalents] == [("A", "O1", "O2"), ("B", "O1", "O2")]
    assert all(atom["operator"] == "x,y,z" for pair in equivalents for atom in pair)

    # --- structure: O1 moved to the O1-O2 midpoint, O2 deleted ---
    st_in = gemmi.read_structure(str(model_path))
    st_out = gemmi.read_structure(str(mmcif_path))
    for chain, resi in CUO_RESIDUES:
        cuo_in = st_in[0][chain][str(resi)][0]
        cuo_out = st_out[0][chain][str(resi)][0]
        assert cuo_out.find_atom("O2", "*") is None
        midpoint = (cuo_in["O1"][0].pos + cuo_in["O2"][0].pos) / 2
        assert cuo_out["O1"][0].pos.dist(midpoint) < 1e-3

    # same atoms (names and positions) as in the tutorial file (B-factors may differ)
    st_tutorial = gemmi.read_structure(str(TUTORIAL / mmcif_path.name))
    atoms_out = {(ch.name, str(res.seqid), a.name): a.pos for ch in st_out[0] for res in ch for a in res}
    atoms_tutorial = {(ch.name, str(res.seqid), a.name): a.pos for ch in st_tutorial[0] for res in ch for a in res}
    assert atoms_out.keys() == atoms_tutorial.keys()
    for key, pos in atoms_out.items():
        assert pos.dist(atoms_tutorial[key]) < 1e-3, key


@pytest.mark.parametrize("equivalents_with_operator", [True, False])
def test_json2restraints_triangles(equivalents_with_operator: bool, tmp_path: Path) -> None:
    """json2restraints.py -e: the bond length restraints of the kept atom (O1) are applied also
    to the deleted atom (O2), angle restraints with these atoms are removed.
    Equivalents without "operator" (from older triangles.py versions) must give the same result.
    """
    equivalents = json.loads((TUTORIAL / "8tnv_nopairs_equivalents.json").read_text(encoding="utf-8"))
    if not equivalents_with_operator:
        for pair in equivalents:
            for atom in pair:
                atom.pop("operator", None)
    equivalents_path = tmp_path / "8tnv_nopairs_equivalents.json"
    equivalents_path.write_text(json.dumps(equivalents), encoding="utf-8")
    json_path = TUTORIAL / "8tnv_nopairs_mc.json"
    model_path = DATA / "models" / "8tnv.cif"
    output_prefix = tmp_path / "8tnv_mc_restraints"

    json2restraints_main(
        jsonPaths=[str(json_path)],
        stPath=str(model_path),
        outputPrefix=str(output_prefix),
        jsonEquivalentsPath=str(equivalents_path),
    )

    txt_path = output_prefix.with_suffix(".txt")
    mmcif_path = output_prefix.with_suffix(".mmcif")
    for path in [txt_path, mmcif_path]:
        assert path.exists(), f"Missing generated file: {path}"

    output_txt = txt_path.read_text(encoding="utf-8")

    # --- Servalcat/Refmac keywords (.txt) ---
    dist_lines = [line.strip() for line in output_txt.splitlines() if line.startswith("exte dist ")]
    angle_lines = [line.strip() for line in output_txt.splitlines() if line.startswith("exte angle ")]
    for line in dist_lines:
        assert re.match(EXTE_DIST_PATTERN, line), f"Unexpected exte dist syntax: {line}"
    for line in angle_lines:
        assert re.match(EXTE_ANGLE_PATTERN, line), f"Unexpected exte angle syntax: {line}"
    assert len(dist_lines) == 28
    assert len(angle_lines) == 12

    # O1 and O2: the same restraints to CU1 and CU2, 2 variants each, no angle restraints
    for chain, resi in CUO_RESIDUES:
        restraints = {}
        for o in ("O1", "O2"):
            atom = f"chain {chain} resi {resi} inse . atom {o} "
            restraints[o] = sorted(line.replace(f"atom {o} ", "atom O ") for line in dist_lines if atom in line)
            assert len(restraints[o]) == 4, f"{chain}{resi} {o}"
            assert not [line for line in angle_lines if atom in line], f"{chain}{resi} {o}"
        assert restraints["O1"] == restraints["O2"]

    # restraints identical to the tutorial result
    tutorial_txt = (TUTORIAL / txt_path.name).read_text(encoding="utf-8")
    tutorial_lines = [line.strip() for line in tutorial_txt.splitlines() if line.startswith("exte ")]
    assert sorted(dist_lines + angle_lines) == sorted(tutorial_lines)

    # --- Servalcat: load the restraints for the output structure, no refinement run ---
    if SERVALCAT_AVAIL:
        st = gemmi.read_structure(str(mmcif_path))
        geom = read_txt_restraints(st, txt_path)
        assert len(geom.bonds) == len(dist_lines)
        assert len(geom.angles) == len(angle_lines)
    else:
        warnings.warn("servalcat not available: Servalcat checks not run")
