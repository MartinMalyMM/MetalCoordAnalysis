"""
Generate external bond length and angle restraints from JSON files from MetalCoord stats to keywords
for Servalcat / Refmacat / Coot / Phenix / Buster.

Typical usage:
ccp4-python json2restraints.py -i 4dl8_AF3_mc.json 4dl8_MG_mc.json 4dl8_NA_mc.json -p 4dl8.cif -o mc_restraints

The script creates these files:

 - mc_restraints.json is a file for Servalcat (version 0.4.150 or later)
   in the JSON format containing the external bond
   length and angle restraints based on the best matching coordination class
   (a grouped-fallback class with centroids is preferred if available).
   Restraints for grouped ligand atoms (e.g. bidentate carboxylate oxygens) are given
   as restraints to their centroid (cdist, cangl).
 - mc_restraints.yaml is a configuration file for Servalcat (version 0.4.150 or later)
   specifying mc_restraints.json as a file with external restraints.
   It can be used as: servalcat refine_xtal_norefmac ... --config mc_restraints.yaml
   Note that the path to mc_restraints.json is relative to the directory where Servalcat is run.
 - mc_restraints.txt is a keyword file for Servalcat or Refmacat containing the external bond
   length and angle restraints based on the best matching coordination class
   (a grouped-fallback class with centroids is preferred if available).
 - mc_restraints_coot.txt is the same keyword file in a simplified format which is compatible with Coot.
   It can be loaded in Coot using Calculate -> Modules -> Restraints and then Restraints -> Read Refmac Extra Restraints.
   The file does not include restraints for atoms with alternative conformations and atoms from symmetry-related molecules.
 - mc_restraints.params is a keyword file compatible with Phenix.refine.
   The file does not include restraints for atoms from symmetry-related molecules.
 - mc_restraints_buster.txt is a file with restraints for Buster (refine -restraints mc_restraints_buster.txt).
   Angle targets and sigmas are given in radians as Buster requires.
   The file does not include restraints for atoms from symmetry-related molecules
   and only one variant of each bond length restraint is included (see --distance-variant):
   by default the variant closest to the distance in the structure model (the first variant if no model is given).

Note that centroid restraints are supported only in the Servalcat JSON format. Therefore, files other than
mc_restraints.json (and mc_restraints.yaml referring to it) do not contain any restraints for grouped ligand
atoms (e.g. bidentate carboxylate oxygens) collapsed to a centroid in a grouped-fallback coordination class.

When a structure model (PDB/mmCIF file) is given, the original metal-related LINK/connection records specified
in the input PDB/mmCIF file (and records referring to atoms not found in the model) are deleted and then
the metal-related links are added again from scratch based on the MetalCoord analysis. To turn off the deletion
step, an extra option --keep-links can be used but a user should be very careful and check the result to avoid
inconsistencies in the connection/link records.
For grouped ligand atoms collapsed to a centroid, a link is created to each atom of the group.
These files are then created:
 - mc_restraints.mmcif
 - mc_restraints.pdb

Dependencies:
 - gemmi

To list all the available options, run:
ccp4-python json2restraints.py -h
"""

from __future__ import annotations

__author__ = "Martin Malý"
__date__ = "2026-11-20"

import json
import math
from typing import Any

try:
    import gemmi
except:
    raise SystemExit("ERROR: module GEMMI not found.\n")

# Records from the MetalCoord JSON: an atom (metal, ligand atom or centroid of grouped atoms)
# and a coordination class (an item of the 'ligands' list)
Atom = dict[str, Any]
CoordinationClass = dict[str, Any]
# Keys identifying an atom in the MetalCoord JSON. "operator" is not included as it was added
# only in recent MetalCoord versions (and it is redundant with "symmetry")
ATOM_ID_KEYS = ("name", "chain", "residue", "sequence", "icode", "altloc", "symmetry")
# Atom specification for external restraints in the Servalcat JSON format
AtomSpec = dict[str, Any]


def servalcat_atom_spec(
    chain: str,
    resi: int | str,
    icode: str | None,
    name: str,
    altloc: str | None = None,
    symm: int = 0,
) -> AtomSpec:
    """Atom specification for external restraints in the Servalcat JSON format.
    Mirrors what servalcat.refmac.refmac_keywords.parse_atom_spec() produces
    from the corresponding Refmac keyword."""
    spec: AtomSpec = {"chain": chain, "resi": int(resi), "names": [name]}
    if icode and icode != ".":
        spec["ins"] = icode
    if altloc:
        spec["alt"] = altloc
    if symm:
        spec["symm"] = True
    return spec


def servalcat_group_specs(atom: Atom) -> list[AtomSpec]:
    """Group of atom specifications for centroid restraints (cdist, cangl) in the Servalcat JSON format.
    A centroid from MetalCoord ("kind": "centroid") gives a group of all its members,
    a regular atom gives a group of one atom.
    Atoms from symmetry-related molecules are marked with "symm": true."""
    members = atom["members"] if atom.get("kind") == "centroid" else [atom]
    return [
        servalcat_atom_spec(m["chain"], m["sequence"], m["icode"], m["name"], m["altloc"], m["symmetry"])
        for m in members
    ]


def servalcat_centroid_restraint(specs: list[list[AtomSpec]], value: float, sigma: float) -> dict[str, Any]:
    """Centroid restraint (cdist, cangl) in the Servalcat JSON format.
    If any atom is from a symmetry-related molecule, the whole restraint is marked with "symm": true
    as Servalcat reads the symmetry flag of centroid restraints at the restraint level."""
    restr: dict[str, Any] = {"specs": specs, "value": value, "sigma": sigma}
    if any(spec.get("symm") for group in specs for spec in group):
        restr["symm"] = True
    return restr


def is_grouped_class(ligand: CoordinationClass) -> bool:
    """True if the coordination class contains grouped ligands collapsed to a centroid,
    i.e. its descriptor includes Q, e.g. @OCT{Ca,O,O,O,O,O,Q}."""
    descriptor = ligand.get("descriptor") or ""
    if "{" in descriptor and "}" in descriptor:
        tokens = descriptor[descriptor.index("{") + 1 : descriptor.rindex("}")].split(",")
        return "Q" in (t.strip() for t in tokens)
    return False


def select_coordination_class(ligands: list[CoordinationClass]) -> CoordinationClass | None:
    """Select the coordination class (`ligands` entry in JSON from MetalCoord)
    to generate restraints from.
    A grouped-fallback class (with Q in the descriptor) is preferred if available,
    otherwise the first listed class (the best matching one) is used.
    Returns None if no coordination class is listed."""
    if not ligands:
        return None
    for ligand in ligands:
        if is_grouped_class(ligand):
            return ligand
    return ligands[0]


def atom_address(st: gemmi.Structure, atom: Atom) -> gemmi.AtomAddress:
    """gemmi.AtomAddress of an atom described in the MetalCoord JSON.
    Atom names are taken from the keys 'metal' (metal atom) or 'name' (ligand atom).
    Without altloc, the first matching atom is taken regardless of altloc."""
    name = atom["metal"] if "metal" in atom else atom["name"]
    seqid = str(atom["sequence"])
    if atom["icode"] and atom["icode"] != ".":
        seqid += atom["icode"]
    chain = st[0][atom["chain"]]
    residue = chain[seqid][atom["residue"]]
    return gemmi.make_address(chain, residue, residue.find_atom(name, atom["altloc"] or "*"))


def closest_variant(st: gemmi.Structure | None, atom1: Atom, atom2: Atom, distances: list[float]) -> int:
    """Index of the distance variant closest to the distance of two atoms (in the same asymmetric unit)
    in the structure model. Returns 0 (the first variant) if the model or an atom is not available."""
    if st is None or len(distances) < 2:
        return 0
    try:
        pos1 = st[0].find_cra(atom_address(st, atom1), ignore_segment=True).atom.pos
        pos2 = st[0].find_cra(atom_address(st, atom2), ignore_segment=True).atom.pos
    except (IndexError, KeyError, ValueError, RuntimeError, AttributeError):
        return 0  # an atom not found in the model
    model_distance = pos1.dist(pos2)
    closest = 0
    for i, distance in enumerate(distances):
        if abs(distance - model_distance) < abs(distances[closest] - model_distance):
            closest = i
    return closest


def add_metal_link(st: gemmi.Structure, atom_metal: Atom, atom_ligand: Atom, name: str) -> bool:
    """Add a MetalC LINK / _struct_conn record between a metal and a ligand atom
    unless the same connection already exists. Returns True if the link was added."""
    con = gemmi.Connection()
    con.name = name
    con.type = gemmi.ConnectionType.MetalC
    con.partner1 = atom_address(st, atom_metal)
    con.partner2 = atom_address(st, atom_ligand)
    for con_existing in st.connections:
        if (
            con_existing.partner1 == con.partner1
            and con_existing.partner2 == con.partner2
        ) or (
            con_existing.partner1 == con.partner2
            and con_existing.partner2 == con.partner1
        ):
            return False
    st.connections.append(con)
    return True


def main(
    jsonPaths: list[str],
    stPath: str = "",
    outputPrefix: str = "restraints",
    jsonEquivalentsPath: str = "",
    keep_links: bool = False,
    distance_variant: int = 0,
) -> None:
    """
    Generate external bond length and angle restraints for
    Servalcat / Refmacat / Coot / Phenix / Buster
    from the JSON files from MetalCoord stats.

    Args:
        jsonPaths (list[str]): List of MetalCoord JSON file names or paths.
                               Required argument.
        stPath (str): PDB/mmCIF file name or path. Required for updating LINK/connection records
                      and writing the output structure files (.mmcif, .pdb).
        outputPrefix (str): Prefix for output file names.
        jsonEquivalentsPath (str): Only for a special case of metal-atom-atom-metal triangles:
                                   JSON file specifying equivalent atoms.
        keep_links (bool): Keep metal-related links in the input stPath file.
                           Default: False
        distance_variant (int): If more variants for the same distance restraint are possible,
                                write only the n-th variant (counted from 1) to the output restraints files
                                for Phenix and Buster. It may help troubleshooting.
                                Default: 0 so all variants will be written for Phenix
                                and the variant closest to the distance in the model for Buster
                                (Buster keeps only one restraint for the same atoms).

    Returns:
        None"""

    outputRestraintsPath = outputPrefix + ".txt"
    outputRestraintsJsonPath = outputPrefix + ".json"
    outputRestraintsYamlPath = outputPrefix + ".yaml"
    outputRestraintsCootPath = outputPrefix + "_coot.txt"
    outputRestraintsPhenixPath = outputPrefix + ".params"
    outputMmcifPath = outputPrefix + ".mmcif"
    outputPdbPath = outputPrefix + ".pdb"

    if stPath:
        st = gemmi.read_structure(stPath)
        if not keep_links:
            # Delete LINK connection records involving metal (usually MetalC type) atoms or atoms not found in any model
            connections_kept = gemmi.ConnectionList()
            for connection in st.connections:
                for m in range(len(st)):
                    partner1_atom = (
                        st[m].find_cra(connection.partner1, ignore_segment=True).atom
                    )
                    partner2_atom = (
                        st[m].find_cra(connection.partner2, ignore_segment=True).atom
                    )
                    if partner1_atom and partner2_atom:  # check if atoms exist
                        if not (
                            partner1_atom.element.is_metal
                            or partner2_atom.element.is_metal
                        ):
                            connections_kept.append(connection)
                        break  # keep each connection only once, even for multiple models
            n_links_old = len(st.connections)
            n_links_kept = len(connections_kept)
            st.connections.clear()
            st.connections = connections_kept
            n_links_deleted = n_links_old - n_links_kept
            print(
                f"Number of deleted links related to metals from the input structure file: {n_links_deleted}"
            )

    # Load JSON files from metalCoord
    d: list[dict[str, Any]] = []  # one record per metal atom
    for jsonPath in jsonPaths:
        with open(jsonPath, "r", encoding="utf-8") as f:
            d.extend(json.load(f))

    # If provided, load a JSON file describing equivalent atoms (mainly for metal-atom-atom-metal triangles)
    # and apply to the JSON information from metalCoord
    # - append records in 'base' and 'pdb' sections for exte bond restraints (template atom -> equivalent atom)
    # - delete records in 'angles' section for exte angle restraints (with the template atom)
    if jsonEquivalentsPath:
        print("JSON file describing equivalent atoms will be applied.")
        atom_pairs_all: list[list[Atom]] = []
        with open(jsonEquivalentsPath, "r", encoding="utf-8") as f:
            atom_pairs_all.extend(json.load(f))
        for atom_pairs in atom_pairs_all:
            for m, atom_metal in enumerate(d):
                for l, ligand in enumerate(atom_metal["ligands"]):
                    for base_or_pdb in ["base", "pdb"]:
                        for j, atom_ligand in enumerate(ligand[base_or_pdb]):
                            if all(  # same atom
                                atom_pairs[0][k] == d[m]["ligands"][l][base_or_pdb][j]["ligand"][k]
                                for k in ATOM_ID_KEYS
                            ):
                                print("Bond length: Template atom:", atom_pairs[0])
                                d[m]["ligands"][l][base_or_pdb].append(
                                    d[m]["ligands"][l][base_or_pdb][j].copy()
                                )
                                d[m]["ligands"][l][base_or_pdb][-1]["ligand"] = (
                                    atom_pairs[1]
                                )
                                print("Bond length: Added atom:   ", atom_pairs[1])
                    angle_restraints_to_delete = []
                    for j, atom_ligands in enumerate(ligand["angles"]):
                        for ligand in ["ligand1", "ligand2"]:
                            if all(  # same atom
                                atom_pairs[0][k] == d[m]["ligands"][l]["angles"][j][ligand][k]
                                for k in ATOM_ID_KEYS
                            ):
                                angle_restraints_to_delete.append(j)
                                print("Bond angle:  Deleted atom: ", atom_pairs[0])
                    if angle_restraints_to_delete:
                        d[m]["ligands"][l]["angles"] = [
                            i
                            for j, i in enumerate(d[m]["ligands"][l]["angles"])
                            if j not in angle_restraints_to_delete
                        ]

    outputLines: list[str] = []
    outputRestraintsJson: list[dict[str, Any]] = []
    outputLinesCoot: list[str] = []
    outputLinesPhenix: list[str] = []
    outputLinesBuster: list[str] = []
    # Buster: add bond|bcor|angle|plane  <target> <sigma> [<weight>] <atom-1> ... <atom-N>
    # Buster: <chain>|<resnum>:<atmnam>[.<altConf>]
    i_con = 1
    outputLines.append("### RESTRAINTS FROM METALCOORD - BEGINNING ###\n")
    outputLines.append("# Note that this file is not compatible with Coot \n")
    outputLinesPhenix.append("refinement.geometry_restraints.edits {\n")
    for atom_metal in d:  # for every metal atom
        if not atom_metal["icode"]:
            atom_metal["icode"] = "."
        if atom_metal["icode"] != ".":
            atom_metal["sequence_icode"] = str(atom_metal["sequence"]) + str(
                atom_metal["icode"]
            )
        else:
            atom_metal["sequence_icode"] = str(atom_metal["sequence"])
        metal_spec = [
            servalcat_atom_spec(
                atom_metal["chain"],
                atom_metal["sequence"],
                atom_metal["icode"],
                atom_metal["metal"],
                atom_metal["altloc"],
            )
        ]
        selected_class = select_coordination_class(atom_metal["ligands"])
        for ligand in [selected_class] if selected_class else []:
            outputLines.append(
                f"# {atom_metal['chain']}{atom_metal['sequence_icode']}/{atom_metal['metal']} {atom_metal['altloc']} {atom_metal['residue']}; class: {ligand['class']}; procrustes: {str(ligand['procrustes'])}; coordination: {str(ligand['coordination'])}; count: {str(ligand['count'])}; description: {str(ligand['description'])}\n"
            )
            # exte dist - 'base' for within ligand, 'pdb' for its neighbourhood
            for j, atom_ligand in enumerate(ligand["base"] + ligand["pdb"]):
                if not atom_ligand["ligand"]["icode"]:
                    atom_ligand["ligand"]["icode"] = "."
                if atom_ligand["ligand"]["icode"] != ".":
                    atom_ligand["ligand"]["sequence_icode"] = str(
                        atom_ligand["ligand"]["sequence"]
                    ) + str(atom_ligand["ligand"]["icode"])
                else:
                    atom_ligand["ligand"]["sequence_icode"] = str(
                        atom_ligand["ligand"]["sequence"]
                    )
                # Buster keeps only one restraint for the same atoms ("add" overwrites it) so only one
                # variant is written: the selected one or the one closest to the distance in the model
                if 1 <= distance_variant <= len(atom_ligand["distance"]):
                    buster_variant = distance_variant - 1
                else:
                    buster_variant = closest_variant(
                        st if stPath else None, atom_metal, atom_ligand["ligand"], atom_ligand["distance"]
                    )
                for i in range(len(atom_ligand["distance"])):
                    line = f"exte dist first chain {atom_metal['chain']} resi {atom_metal['sequence']} inse {atom_metal['icode']} atom {atom_metal['metal']} "
                    line_coot = line
                    line_buster = f"add bond {atom_ligand['distance'][i]} {atom_ligand['std'][i]} "
                    line_buster += f"{atom_metal['chain']}|{atom_metal['sequence_icode']}:{atom_metal['metal']}"
                    atom_selection_1_phenix = f"chain {atom_metal['chain']} and resname {atom_metal['residue']} and resid {atom_metal['sequence_icode']} and name {atom_metal['metal']}"
                    if atom_metal["altloc"]:  # not for Coot
                        line += f"altecode {atom_metal['altloc']} "
                        line_buster += f".{atom_metal['altloc']}"
                        atom_selection_1_phenix += f" and altloc {atom_metal['altloc']}"
                    line += f"second chain {atom_ligand['ligand']['chain']} resi {atom_ligand['ligand']['sequence']} inse {atom_ligand['ligand']['icode']} atom {atom_ligand['ligand']['name']} "
                    line_coot += f"second chain {atom_ligand['ligand']['chain']} resi {atom_ligand['ligand']['sequence']} inse {atom_ligand['ligand']['icode']} atom {atom_ligand['ligand']['name']} "
                    line_buster += f" {atom_ligand['ligand']['chain']}|{atom_ligand['ligand']['sequence_icode']}:{atom_ligand['ligand']['name']}"
                    atom_selection_2_phenix = f"chain {atom_ligand['ligand']['chain']} and resname {atom_ligand['ligand']['residue']} and resid {atom_ligand['ligand']['sequence_icode']} and name {atom_ligand['ligand']['name']}"
                    if atom_ligand["ligand"]["altloc"]:  # not for Coot
                        line += f"altecode {atom_ligand['ligand']['altloc']} "
                        line_buster += f".{atom_ligand['ligand']['altloc']}"
                        atom_selection_2_phenix += (
                            f" and altloc {atom_ligand['ligand']['altloc']}"
                        )
                    line += f"value {atom_ligand['distance'][i]} sigma {atom_ligand['std'][i]}"
                    line_coot += f"value {atom_ligand['distance'][i]} sigma {atom_ligand['std'][i]}"
                    if atom_ligand["ligand"]["symmetry"]:
                        line += " symm y"
                    if i == 0:
                        line += f" type 0"
                    else:
                        line += f" type 1"
                    if len(atom_ligand["distance"]) >= 2:
                        line += f" # variant {str(i + 1)}"
                    # print(line)
                    outputLines.append(line + "\n")
                    restr_json = {
                        "specs": [
                            metal_spec,
                            [
                                servalcat_atom_spec(
                                    atom_ligand["ligand"]["chain"],
                                    atom_ligand["ligand"]["sequence"],
                                    atom_ligand["ligand"]["icode"],
                                    atom_ligand["ligand"]["name"],
                                    atom_ligand["ligand"]["altloc"],
                                )
                            ],
                        ],
                        "value": atom_ligand["distance"][i],
                        "sigma": atom_ligand["std"][i],
                    }
                    if atom_ligand["ligand"]["symmetry"]:
                        restr_json["symm"] = True
                    restr_json["type"] = 0 if i == 0 else 1
                    outputRestraintsJson.append({"rest_type": "dist", "restr": restr_json})
                    # restraints for Coot cannot include atoms with altloc and symmetry identifiers
                    if (
                        not atom_metal["altloc"]
                        and not atom_ligand["ligand"]["altloc"]
                        and not atom_ligand["ligand"]["symmetry"]
                    ):
                        outputLinesCoot.append(line_coot + "\n")
                    # restraints for Buster: atoms from symmetry-related molecules cannot be specified
                    if i == buster_variant and not atom_ligand["ligand"]["symmetry"]:
                        outputLinesBuster.append(line_buster + "\n")
                    if (
                        len(atom_ligand["distance"]) < 2
                        or not distance_variant
                        or (
                            len(atom_ligand["distance"]) >= 2
                            and i + 1 == distance_variant
                        )
                    ):
                        # restraints for phenix.refine: atoms with symmetry records are not included
                        if not atom_ligand["ligand"]["symmetry"]:
                            outputLinesPhenix.append("  bond {\n")
                            outputLinesPhenix.append("    action = *add\n")
                            outputLinesPhenix.append(
                                f"    atom_selection_1 = {atom_selection_1_phenix}\n"
                            )
                            outputLinesPhenix.append(
                                f"    atom_selection_2 = {atom_selection_2_phenix}\n"
                            )
                            outputLinesPhenix.append(
                                f"    distance_ideal = {atom_ligand['distance'][i]}\n"
                            )
                            outputLinesPhenix.append(
                                f"    sigma = {atom_ligand['std'][i]}\n"
                            )
                            outputLinesPhenix.append("  }\n")
                    if j >= len(ligand["base"]) and stPath:
                        # create a LINK /_struct_conn in PDB-mmCIF, it's from 'pdb', i.e. neighbourhood
                        if add_metal_link(
                            st, atom_metal, atom_ligand["ligand"], "metalcoord" + str(i_con)
                        ):
                            i_con += 1
            # exte angle
            for atom_ligands in ligand[
                "angles"
            ]:  # if any angles specified in ligand['angles']
                if not atom_ligands["ligand1"]["icode"]:
                    atom_ligands["ligand1"]["icode"] = "."
                if not atom_ligands["ligand2"]["icode"]:
                    atom_ligands["ligand2"]["icode"] = "."
                if atom_ligands["ligand1"]["icode"] != ".":
                    atom_ligands["ligand1"]["sequence_icode"] = str(
                        atom_ligands["ligand1"]["sequence"]
                    ) + str(atom_ligands["ligand1"]["icode"])
                else:
                    atom_ligands["ligand1"]["sequence_icode"] = str(
                        atom_ligands["ligand1"]["sequence"]
                    )
                if atom_ligands["ligand2"]["icode"] != ".":
                    atom_ligands["ligand2"]["sequence_icode"] = str(
                        atom_ligands["ligand2"]["sequence"]
                    ) + str(atom_ligands["ligand2"]["icode"])
                else:
                    atom_ligands["ligand2"]["sequence_icode"] = str(
                        atom_ligands["ligand2"]["sequence"]
                    )
                line = f"exte angle "
                # ligand1
                line += f"first chain {atom_ligands['ligand1']['chain']} resi {atom_ligands['ligand1']['sequence']} inse {atom_ligands['ligand1']['icode']} atom {atom_ligands['ligand1']['name']} "
                line_coot = line
                # Buster reads angle targets and sigmas in radians
                line_buster = f"add angle {math.radians(atom_ligands['angle']):.6f} {math.radians(atom_ligands['std']):.6f} "
                line_buster += f"{atom_ligands['ligand1']['chain']}|{atom_ligands['ligand1']['sequence_icode']}:{atom_ligands['ligand1']['name']}"
                atom_selection_1_phenix = f"chain {atom_ligands['ligand1']['chain']} and resname {atom_ligands['ligand1']['residue']} and resid {atom_ligands['ligand1']['sequence_icode']} and name {atom_ligands['ligand1']['name']}"
                if atom_ligands["ligand1"]["altloc"]:
                    line += f"altecode {atom_ligands['ligand1']['altloc']} "
                    line_buster += f".{atom_ligands['ligand1']['altloc']}"
                    atom_selection_1_phenix += (
                        f" and altloc {atom_ligands['ligand1']['altloc']}"
                    )
                if atom_ligands["ligand1"]["symmetry"]:
                    line += "symm y "
                # metal
                line += f"next chain {atom_metal['chain']} resi {atom_metal['sequence']} inse {atom_metal['icode']} atom {atom_metal['metal']} "
                line_coot += f"next chain {atom_metal['chain']} resi {atom_metal['sequence']} inse {atom_metal['icode']} atom {atom_metal['metal']} "
                line_buster += f" {atom_metal['chain']}|{atom_metal['sequence_icode']}:{atom_metal['metal']}"
                atom_selection_2_phenix = f"chain {atom_metal['chain']} and resname {atom_metal['residue']} and resid {atom_metal['sequence_icode']} and name {atom_metal['metal']}"
                if atom_metal["altloc"]:
                    line += f"altecode {atom_metal['altloc']} "
                    line_buster += f".{atom_metal['altloc']}"
                    atom_selection_2_phenix += f" and altloc {atom_metal['altloc']}"
                # ligand2
                line += f"next chain {atom_ligands['ligand2']['chain']} resi {atom_ligands['ligand2']['sequence']} inse {atom_ligands['ligand2']['icode']} atom {atom_ligands['ligand2']['name']} "
                line_coot += f"next chain {atom_ligands['ligand2']['chain']} resi {atom_ligands['ligand2']['sequence']} inse {atom_ligands['ligand2']['icode']} atom {atom_ligands['ligand2']['name']} "
                line_buster += f" {atom_ligands['ligand2']['chain']}|{atom_ligands['ligand2']['sequence_icode']}:{atom_ligands['ligand2']['name']}"
                atom_selection_3_phenix = f"chain {atom_ligands['ligand2']['chain']} and resname {atom_ligands['ligand2']['residue']} and resid {atom_ligands['ligand2']['sequence_icode']} and name {atom_ligands['ligand2']['name']}"
                if atom_ligands["ligand2"]["altloc"]:
                    line += f"altecode {atom_ligands['ligand2']['altloc']} "
                    line_buster += f".{atom_ligands['ligand2']['altloc']}"
                    atom_selection_3_phenix += (
                        f" and altloc {atom_ligands['ligand2']['altloc']}"
                    )
                if atom_ligands["ligand2"]["symmetry"]:
                    line += "symm y "
                line += f"value {round(atom_ligands['angle'], 2)} sigma {round(atom_ligands['std'], 2)}"
                line_coot += f"value {round(atom_ligands['angle'], 2)} sigma {round(atom_ligands['std'], 2)}"
                line += f" type 0"
                # print(line)
                outputLines.append(line + "\n")
                outputRestraintsJson.append(
                    {
                        "rest_type": "angl",
                        "restr": {
                            "specs": [
                                [
                                    servalcat_atom_spec(
                                        atom_ligands["ligand1"]["chain"],
                                        atom_ligands["ligand1"]["sequence"],
                                        atom_ligands["ligand1"]["icode"],
                                        atom_ligands["ligand1"]["name"],
                                        atom_ligands["ligand1"]["altloc"],
                                        atom_ligands["ligand1"]["symmetry"],
                                    )
                                ],
                                metal_spec,
                                [
                                    servalcat_atom_spec(
                                        atom_ligands["ligand2"]["chain"],
                                        atom_ligands["ligand2"]["sequence"],
                                        atom_ligands["ligand2"]["icode"],
                                        atom_ligands["ligand2"]["name"],
                                        atom_ligands["ligand2"]["altloc"],
                                        atom_ligands["ligand2"]["symmetry"],
                                    )
                                ],
                            ],
                            "value": round(atom_ligands["angle"], 2),
                            "sigma": round(atom_ligands["std"], 2),
                            "type": 0,
                        },
                    }
                )
                # restraints for Buster: atoms from symmetry-related molecules cannot be specified
                if not atom_ligands["ligand1"]["symmetry"] and not atom_ligands["ligand2"]["symmetry"]:
                    outputLinesBuster.append(line_buster + "\n")
                # restraints for Coot cannot include atoms with altloc and symmetry identifiers
                if (
                    not atom_metal["altloc"]
                    and not atom_ligands["ligand1"]["altloc"]
                    and not atom_ligands["ligand1"]["symmetry"]
                    and not atom_ligands["ligand2"]["altloc"]
                    and not atom_ligands["ligand2"]["symmetry"]
                ):
                    outputLinesCoot.append(line_coot + "\n")
                # restraints for phenix.refine cannot include atoms with symmetry
                if (
                    not atom_ligands["ligand1"]["symmetry"]
                    and not atom_ligands["ligand2"]["symmetry"]
                ):
                    outputLinesPhenix.append("  angle {\n")
                    if (
                        atom_metal["chain"] == atom_ligands["ligand1"]["chain"]
                        and atom_metal["chain"] == atom_ligands["ligand2"]["chain"]
                        and atom_metal["residue"] == atom_ligands["ligand1"]["residue"]
                        and atom_metal["residue"] == atom_ligands["ligand2"]["residue"]
                        and atom_metal["sequence_icode"]
                        == atom_ligands["ligand1"]["sequence_icode"]
                        and atom_metal["sequence_icode"]
                        == atom_ligands["ligand2"]["sequence_icode"]
                    ):
                        outputLinesPhenix.append("    action = *change\n")
                    else:
                        outputLinesPhenix.append("    action = *add\n")
                    outputLinesPhenix.append(
                        f"    atom_selection_1 = {atom_selection_1_phenix}\n"
                    )
                    outputLinesPhenix.append(
                        f"    atom_selection_2 = {atom_selection_2_phenix}\n"
                    )
                    outputLinesPhenix.append(
                        f"    atom_selection_3 = {atom_selection_3_phenix}\n"
                    )
                    outputLinesPhenix.append(
                        f"    angle_ideal = {round(atom_ligands['angle'], 2)}\n"
                    )
                    outputLinesPhenix.append(
                        f"    sigma = {round(atom_ligands['std'], 2)}\n"
                    )
                    outputLinesPhenix.append("  }\n")
            # centroid restraints for grouped ligands (e.g. bidentate carboxylates)
            # only for Servalcat JSON - cannot be expressed in the other formats
            for grouped in ligand.get("grouped_base", []):
                if stPath and grouped["ligand"].get("kind") == "centroid":
                    # LINK /_struct_conn in PDB-mmCIF to each member atom of the centroid
                    for member in grouped["ligand"]["members"]:
                        if add_metal_link(st, atom_metal, member, "metalcoord" + str(i_con)):
                            i_con += 1
                outputRestraintsJson.append(
                    {
                        "rest_type": "cdist",
                        "restr": servalcat_centroid_restraint(
                            [metal_spec, servalcat_group_specs(grouped["ligand"])],
                            grouped["distance"],
                            grouped["std"],
                        ),
                    }
                )
            for grouped in ligand.get("grouped_angles", []):
                outputRestraintsJson.append(
                    {
                        "rest_type": "cangl",
                        "restr": servalcat_centroid_restraint(
                            [
                                servalcat_group_specs(grouped["ligand1"]),
                                metal_spec,
                                servalcat_group_specs(grouped["ligand2"]),
                            ],
                            round(grouped["angle"], 2),
                            round(grouped["std"], 2),
                        ),
                    }
                )
    outputLines.append("### RESTRAINTS FROM METALCOORD - END ###\n")
    outputLinesPhenix.append("}\n")
    with open(outputRestraintsPath, "w", encoding="utf-8") as f:
        f.writelines(outputLines)
    # Servalcat JSON format
    with open(outputRestraintsJsonPath, "w", encoding="utf-8") as f:
        f.write("[\n")
        f.write(",\n".join(json.dumps(r) for r in outputRestraintsJson))
        f.write("\n]\n")
    # Servalcat configuration file (--config) pointing to the JSON restraints file
    with open(outputRestraintsYamlPath, "w", encoding="utf-8") as f:
        f.write("refine:\n")
        f.write("  exte_files:\n")
        f.write(f"    - {json.dumps(outputRestraintsJsonPath)}\n")
    with open(outputRestraintsCootPath, "w", encoding="utf-8") as f:
        f.writelines(outputLinesCoot)
    with open(outputPrefix + "_buster.txt", "w", encoding="utf-8") as f:
        f.writelines(outputLinesBuster)
    with open(outputRestraintsPhenixPath, "w", encoding="utf-8") as f:
        f.writelines(outputLinesPhenix)
    if stPath:
        if i_con == 1:
            print("No link added into the input structure file.")
        else:
            print(f"Number of added links in the input structure file: {str(i_con-1)}")
        st.make_mmcif_document().write_file(outputMmcifPath)
        try:
            st.write_pdb(outputPdbPath)
        except Exception as e:
            import sys
            import os

            sys.stderr.write("WARNING: Output PDB file could not be written: ")
            sys.stderr.write(str(e))
            sys.stderr.write("\n")
            try:
                if os.path.exists(outputPdbPath):
                    os.remove(outputPdbPath)
            except:
                pass
    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-i",
        help="JSON file(s) from metalCoord",
        type=str,
        required=True,
        nargs="+",
        metavar="JSON_FILE",
    )
    parser.add_argument(
        "-p",
        help="PDB or mmCIF file; its LINK records are updated in OUTPUT.mmcif and OUTPUT.pdb",
        type=str,
        default="",
        metavar="STRUCTURE",
    )
    parser.add_argument(
        "-o",
        help="Prefix for output file names",
        type=str,
        default="restraints",
        metavar="OUTPUT",
    )
    parser.add_argument(
        "--keep-links",
        help="Keep link connection records related to metals in the input structure file",
        action="store_true",
    )
    parser.add_argument(
        "-e",
        help=(
            "JSON file describing equivalent atoms (*_nopairs_equivalents.json from triangles.py),"
            " only for a special case of metal-atom-atom-metal triangles, e.g. Cu2O2 cluster in CUO,"
            " see tutorial.rst. It is a list of atom pairs [template_atom, equivalent_atom],"
            " each atom given as in the MetalCoord JSON files (name, element, chain, residue,"
            " sequence, icode, altloc, symmetry). For every pair, each bond length restraint"
            " to the template atom is copied for the equivalent atom and all angle restraints"
            " including the template atom are deleted."
        ),
        type=str,
        default="",
        metavar="JSON_EQUIVALENTS",
    )
    parser.add_argument(
        "--distance-variant",
        help=(
            "If more variants for the same distance restraint are possible, which variant of a restraint"
            " should be written in restraints for Phenix and Buster, e.g. 1 will select the first variant,"
            " 2 will select the second variant, etc. (Default: 0 so all variants will be written for Phenix"
            " and the variant closest to the distance in the model for Buster"
            " as Buster keeps only one restraint for the same atoms)."
        ),
        type=int,
        default=0,
        metavar="DISTANCE_VARIANT",
    )

    args = parser.parse_args()
    jsonPaths = args.i
    stPath = args.p
    outputPrefix = args.o
    jsonEquivalentsPath = args.e
    keep_links = args.keep_links
    distance_variant = args.distance_variant

    main(
        jsonPaths, stPath, outputPrefix, jsonEquivalentsPath, keep_links, distance_variant
    )
