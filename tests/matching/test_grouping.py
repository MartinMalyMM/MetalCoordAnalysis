import os
import gemmi
import numpy as np
import pytest

from metalCoord.analysis.grouping import create_grouped_view
from metalCoord.analysis.structures import Atom, CifAtom, Ligand
from metalCoord.cif.utils import (
    ATOM_ID,
    ATOM_ID_1,
    ATOM_ID_2,
    TYPE_SYMBOL,
    get_nonmetal_bonds,
)


def _adjacency(*edges):
    graph = {}
    for atom1, atom2 in edges:
        graph.setdefault(atom1, []).append(atom2)
        graph.setdefault(atom2, []).append(atom1)
    return graph


def _atom(name, element, xyz, altloc=""):
    atom = gemmi.Atom()
    atom.name = name
    atom.element = gemmi.Element(element)
    atom.pos = gemmi.Position(*xyz)
    atom.occ = 1
    atom.b_iso = 10
    atom.altloc = altloc if altloc else "\x00"
    return atom


def _cif_ligand(atom_specs, bonds):
    chain = gemmi.Chain("A")
    residue = gemmi.Residue()
    residue.name = "LIG"
    residue.seqid = gemmi.SeqId("1")

    ligand = Ligand(CifAtom(_atom("CU", "Cu", (0, 0, 0)), residue, chain))
    ligand.nonmetal_bonds = bonds
    for name, element, xyz, altloc in atom_specs:
        ligand.add_ligand(CifAtom(_atom(name, element, xyz, altloc), residue, chain))
    return ligand


def test_get_nonmetal_bonds_excludes_metal_edges():
    atoms = {
        ATOM_ID: ["CU1", "CU2", "N1", "C1", "O1"],
        TYPE_SYMBOL: ["Cu", "Cu", "N", "C", "O"],
    }
    bonds = {
        ATOM_ID_1: ["CU1", "CU1", "N1", "N1", "CU1"],
        ATOM_ID_2: ["N1", "CU2", "C1", "O1", "CU2"],
    }

    assert get_nonmetal_bonds(atoms, bonds) == {
        "N1": ["C1", "O1"],
        "C1": ["N1"],
        "O1": ["N1"],
    }


def test_cif_grouping_accepts_paths_up_to_three_edges():
    ligand = _cif_ligand(
        [
            ("N1", "N", (1, 0, 0), ""),
            ("O1", "O", (3, 0, 0), ""),
        ],
        _adjacency(("N1", "C1"), ("C1", "C2"), ("C2", "O1")),
    )

    view = create_grouped_view(ligand)

    assert view is not None
    assert view.coordination() == 1
    assert [atom.atom.name for atom in view.groups[0]] == ["N1", "O1"]
    np.testing.assert_allclose(view.get_coord()[1], [2, 0, 0])


def test_cif_grouping_rejects_paths_longer_than_three_edges():
    ligand = _cif_ligand(
        [
            ("N1", "N", (1, 0, 0), ""),
            ("O1", "O", (5, 0, 0), ""),
        ],
        _adjacency(("N1", "C1"), ("C1", "C2"), ("C2", "C3"), ("C3", "O1")),
    )

    assert create_grouped_view(ligand) is None


def test_cif_grouping_can_collapse_three_ligands_into_one_centroid():
    ligand = _cif_ligand(
        [
            ("N1", "N", (1, 0, 0), ""),
            ("N2", "N", (2, 0, 0), ""),
            ("N3", "N", (3, 0, 0), ""),
        ],
        _adjacency(("N1", "C1"), ("C1", "N2"), ("N2", "C2"), ("C2", "N3")),
    )

    view = create_grouped_view(ligand)

    assert view is not None
    assert view.coordination() == 1
    assert [atom.atom.name for atom in view.groups[0]] == ["N1", "N2", "N3"]
    np.testing.assert_allclose(view.get_coord()[1], [2, 0, 0])


def test_cif_grouping_keeps_incompatible_altlocs_separate():
    ligand = _cif_ligand(
        [
            ("N1", "N", (1, 0, 0), "A"),
            ("N2", "N", (2, 0, 0), "B"),
        ],
        _adjacency(("N1", "N2")),
    )

    assert create_grouped_view(ligand) is None


@pytest.mark.parametrize(
    "altloc1, altloc2, grouped",
    [
        ("", "A", True),
        ("", "", True),
        ("A", "A", True),
        ("B", "B", True),
        ("A", "B", False),
    ],
)
def test_cif_grouping_requires_compatible_altlocs(altloc1, altloc2, grouped):
    ligand = _cif_ligand(
        [
            ("O1", "O", (1, 0, 0), altloc1),
            ("O2", "O", (3, 0, 0), altloc2),
        ],
        _adjacency(("O1", "C1"), ("C1", "O2")),
    )

    view = create_grouped_view(ligand)

    if grouped:
        assert view is not None
        assert [atom.atom.name for atom in view.groups[0]] == ["O1", "O2"]
    else:
        assert view is None


def test_model_grouping_uses_coordinate_bond_graph_with_helpers():
    st = gemmi.Structure()
    st.cell = gemmi.UnitCell(20, 20, 20, 90, 90, 90)
    model = gemmi.Model("1")
    chain = gemmi.Chain("A")
    residue = gemmi.Residue()
    residue.name = "LIG"
    residue.seqid = gemmi.SeqId("1")

    metal_atom = _atom("CU", "Cu", (0, 0, 0))
    ligand1 = _atom("N1", "N", (1.0, 0, 0))
    helper = _atom("C1", "C", (2.4, 0, 0))
    ligand2 = _atom("N2", "N", (3.8, 0, 0))
    for atom in (metal_atom, ligand1, helper, ligand2):
        residue.add_atom(atom)
    chain.add_residue(residue)
    model.add_chain(chain)
    st.add_model(model)

    ligand = Ligand(Atom(metal_atom, residue, chain, None, st))
    ligand.add_ligand(Atom(ligand1, residue, chain, None, st))
    ligand.add_ligand(Atom(ligand2, residue, chain, None, st))

    view = create_grouped_view(ligand)

    assert view is not None
    assert view.coordination() == 1
    assert [atom.atom.name for atom in view.groups[0]] == ["N1", "N2"]


def test_model_grouping_never_joins_interacting_atoms_of_different_residues():
    # 2IXT A/CA 1313: one water modelled as two waters without altlocs
    # (HOH 2238, occ 0.7 and HOH 2243, occ 0.3, 1.2 A apart). The distance-based
    # model graph links them, but they are two residues and must stay separate.
    st = gemmi.Structure()
    st.cell = gemmi.UnitCell(20, 20, 20, 90, 90, 90)
    model = gemmi.Model("1")
    chain = gemmi.Chain("A")
    for name, seqid, atom in (
        ("CA", "1313", _atom("CA", "Ca", (0, 0, 0))),
        ("HOH", "2238", _atom("O", "O", (2.3, 0, 0))),
        ("HOH", "2243", _atom("O", "O", (2.3, 1.2, 0))),
    ):
        residue = gemmi.Residue()
        residue.name = name
        residue.seqid = gemmi.SeqId(seqid)
        residue.add_atom(atom)
        chain.add_residue(residue)
    model.add_chain(chain)
    st.add_model(model)
    chain = st[0]["A"]

    ligand = Ligand(Atom(chain[0][0], chain[0], chain, None, st))
    ligand.add_ligand(Atom(chain[1][0], chain[1], chain, None, st))
    ligand.add_ligand(Atom(chain[2][0], chain[2], chain, None, st))

    assert create_grouped_view(ligand) is None


class _SymmetryCifAtom(CifAtom):
    @property
    def symmetry(self):
        return 1

    @property
    def symmetry_operator(self):
        return "-x,y+1/2,-z"


def _cif_ligand_with_copies(atom_specs, bonds):
    """Like `_cif_ligand`, but each spec says whether the atom is a symmetry copy."""
    chain = gemmi.Chain("A")
    residue = gemmi.Residue()
    residue.name = "LIG"
    residue.seqid = gemmi.SeqId("1")

    ligand = Ligand(CifAtom(_atom("CU", "Cu", (0, 0, 0)), residue, chain))
    ligand.nonmetal_bonds = bonds
    for name, element, xyz, is_symmetry in atom_specs:
        atom_class = _SymmetryCifAtom if is_symmetry else CifAtom
        ligand.add_ligand(atom_class(_atom(name, element, xyz), residue, chain))
    return ligand


def test_grouping_collapses_interacting_atoms_of_one_symmetry_copy():
    ligand = _cif_ligand_with_copies(
        [
            ("O1", "O", (1, 0, 0), True),
            ("O2", "O", (3, 0, 0), True),
        ],
        _adjacency(("O1", "C1"), ("C1", "O2")),
    )

    view = create_grouped_view(ligand)

    assert view is not None
    assert view.coordination() == 1
    centroid = next(iter(view.all_ligands))
    assert [atom.atom.name for atom in centroid.members] == ["O1", "O2"]
    assert centroid.symmetry == 1
    assert centroid.symmetry_operator == "-x,y+1/2,-z"


def test_grouping_never_joins_interacting_atoms_across_crystal_copies():
    ligand = _cif_ligand_with_copies(
        [
            ("O1", "O", (1, 0, 0), False),
            ("O2", "O", (3, 0, 0), True),
        ],
        _adjacency(("O1", "C1"), ("C1", "O2")),
    )

    assert create_grouped_view(ligand) is None


def _move_to_symmetry_mate(st, seqnum, op_triplet):
    """Moves the CA metal `seqnum` by the symmetry operation `op_triplet`."""
    op = gemmi.Op(op_triplet)
    for chain in st[0]:
        for residue in chain:
            if residue.name == "CA" and residue.seqid.num == seqnum:
                for atom in residue:
                    frac = st.cell.fractionalize(atom.pos)
                    moved = op.apply_to_xyz([frac.x, frac.y, frac.z])
                    atom.pos = st.cell.orthogonalize(gemmi.Fractional(*moved))


def _model_site_groups(seqnum, symmetry_op=None):
    from metalCoord.analysis.structures import get_ligands

    path = os.path.join(os.path.dirname(__file__), "..", "data", "models", "3x34.pdb.gz")
    st = gemmi.read_structure(path)
    if symmetry_op:
        _move_to_symmetry_mate(st, seqnum, symmetry_op)
    st.setup_entities()
    ligands, _ = get_ligands(st, "CA")
    (site,) = [l for l in ligands if l.residue.seqid.num == seqnum]
    site.filter_apex_donors()
    view = create_grouped_view(site)
    assert view is not None
    return site, sorted(
        (
            tuple(
                sorted(
                    (m.residue.name, m.residue.seqid.num, m.atom.name)
                    for m in getattr(atom, "members", [atom])
                )
            ),
            atom.symmetry != 0,
        )
        for atom in view.all_ligands
    )


def test_model_grouping_is_invariant_to_moving_the_metal_to_a_symmetry_mate():
    # Moving CA A102 of 3x34 onto a symmetry mate swaps its interacting atoms
    # between the asymmetric unit and a symmetry copy, but the carboxylate
    # groups (GLU 42, GLU 48) must collapse exactly the same way.
    site, groups = _model_site_groups(102)
    site_symm, groups_symm = _model_site_groups(102, "-x+2,y-1/2,-z-1/2")

    assert site.coordination() == site_symm.coordination() == 8
    assert [members for members, _ in groups] == [
        members for members, _ in groups_symm
    ]
    for (members, is_symm), (_, is_symm_edit) in zip(groups, groups_symm):
        assert is_symm != is_symm_edit
    carboxylates = [m for m, _ in groups_symm if len(m) > 1]
    assert carboxylates == [
        (("GLU", 42, "OE1"), ("GLU", 42, "OE2")),
        (("GLU", 48, "OE1"), ("GLU", 48, "OE2")),
    ]
