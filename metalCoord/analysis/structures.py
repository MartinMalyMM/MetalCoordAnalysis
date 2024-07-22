from operator import le
import re
from unittest import result
from matplotlib.pylab import f
from networkx import all_neighbors
import numpy as np
import gemmi
from sklearn import neighbors
from metalCoord.config import Config


class Atom:
    """
    Represents an atom in a molecular structure.

    Attributes:
        atom (str): The name of the atom.
        residue (str): The name of the residue the atom belongs to.
        chain (str): The chain identifier of the atom.
    """

    def __init__(self, atom, residue, chain):
        self._atom = atom
        self._residue = residue
        self._chain = chain

    @property
    def atom(self):
        """
        Get the atom associated with this structure.

        Returns:
            The atom object associated with this structure.
        """
        return self._atom

    @property
    def residue(self):
        """
        Returns the residue associated with the structure.

        Returns:
            The residue object associated with the structure.
        """
        return self._residue

    @property
    def chain(self):
        """
        Returns the chain associated with the structure.

        Returns:
            str: The chain identifier.
        """
        return self._chain


class Ligand:
    """
    Represents a ligand associated with a metal in a structure.

    Attributes:
        _metal (gemmi.Atom): The metal atom.
        _residue (gemmi.Residue): The residue containing the metal.
        _chain (gemmi.Chain): The chain containing the residue.
        _ligands (list): List of atoms from ligand associated with the metal.
        _extra_ligands (list): List of additional atom not from ligands associated with the metal.
    """

    def __init__(self, metal, residue, chain) -> None:
        self._metal = metal
        self._residue = residue
        self._chain = chain
        self._ligands = []
        self._extra_ligands = []

    def clean_the_farthest(self, free: bool = False) -> "Ligand":
        """
        Cleans the farthest ligand from the metal coordination.

        Args:
            free (bool, optional): If True, includes ligands from both self._ligands and self._extra_ligands.
                                   If False, includes only ligands from self._extra_ligands. Default is False.

        Returns
            Ligand: A new Ligand object with the farthest ligand removed.

        """
        atoms = self._ligands + self._extra_ligands if free else self._extra_ligands
        to_delete = list(sorted([[l.atom, self._cov_dist_coeff(l)]
                         for l in atoms], key=lambda x: x[1], reverse=True))[0][0]
        cleaned_ligand = Ligand(self._metal, self._residue, self._chain)

        for l in self._ligands:
            if free:
                if l.atom != to_delete:
                    cleaned_ligand.add_ligand(l)
            else:
                cleaned_ligand.add_ligand(l)
        for l in self._extra_ligands:
            if l.atom != to_delete:
                cleaned_ligand.add_extra_ligand(l)
        return cleaned_ligand

    def _euclidean(self, atom1: gemmi.Atom, atom2: gemmi.Atom):
        """
        Calculates the Euclidean distance between two atoms.

        Args:
            atom1 (gemmi.Atom): The first atom.
            atom2 (gemmi.Atom): The second atom.

        Returns:
            The Euclidean distance between the two atoms.
        """
        return np.sqrt((atom1.pos - atom2.pos).dot(atom1.pos - atom2.pos))

    def _dist(self, l: Atom):
        """
        Calculates the distance between the metal and a ligand atom.

        Args:
            l (Atom): The ligand atom.

        Returns:
            The distance between the metal and the ligand atom.
        """
        return self._euclidean(self.metal, l.atom)

    def _cov_dist(self, l: Atom):
        """
        Calculates the covalent distance between the metal and a ligand atom.

        Args:
            l (Atom): The ligand atom.

        Returns:
            The covalent distance between the metal and the ligand atom.
        """
        return gemmi.Element(self.metal.element.name).covalent_r + gemmi.Element(l.atom.element.name).covalent_r

    def _cov_dist_coeff(self, l: Atom):
        """
        Calculates the covalent distance coefficient between the metal and a ligand atom.

        Args:
            l (Atom): The ligand atom.

        Returns:
            The covalent distance coefficient between the metal and the ligand atom.
        """
        return self._euclidean(self.metal, l.atom) / self._cov_dist(l)

    def cov_dist_coeffs(self):
        """
        Calculates the covalent distance coefficients for all extra ligands.

        Returns:
            A dictionary containing the covalent distance coefficients for each extra ligand.
        """
        return {l.atom.name: self._cov_dist_coeff(l) for l in self._extra_ligands}

    @property
    def metal(self):
        """
        Returns the metal associated with the structure.

        Returns:
            The metal element.
        """
        return self._metal

    @property
    def residue(self):
        """
        Returns the residue associated with the structure.

        Returns:
            The residue object associated with the structure.
        """
        return self._residue

    @property
    def chain(self):
        """
        Returns the chain associated with the structure.

        Returns:
            str: The chain associated with the structure.
        """
        return self._chain

    @property
    def ligands(self):
        """
        Generator that yields each ligand associated with the metal.

        Yields:
            Each ligand atom.
        """
        for ligand in self._ligands:
            yield ligand

    @property
    def extra_ligands(self):
        """
        Generator that yields each extra ligand associated with the metal.

        Yields:
            Each extra ligand atom.
        """
        for ligand in self._extra_ligands:
            yield ligand

    @property
    def all_ligands(self):
        """
        Generator that yields each ligand and extra ligand associated with the metal.

        Yields:
            Each ligand and extra ligand atom.
        """
        for ligand in self._ligands + self._extra_ligands:
            yield ligand

    @property
    def ligands_len(self):
        """
        Returns the number of ligands associated with the metal.

        Returns:
            The number of ligands.
        """
        return len(self._ligands)

    def add_ligand(self, ligand: Atom):
        """
        Adds a ligand to the structure.

        Args:
            ligand (Atom): The ligand to add.
        """
        self._ligands.append(ligand)

    def add_extra_ligand(self, ligand: Atom):
        """
        Adds an extra ligand to the structure.

        Args:
            ligand (Atom): The extra ligand to add.
        """
        self._extra_ligands.append(ligand)

    def elements(self):
        """
        Returns a sorted list of element names for all ligands and extra ligands.

        Returns:
            A sorted list of element names.
        """
        return sorted([ligand.atom.element.name for ligand in self._ligands + self._extra_ligands])

    def atoms(self):
        """
        Returns a list of element names for all ligands and extra ligands.

        Returns:
            A list of element names.
        """
        return [ligand.atom.element.name for ligand in self._ligands + self._extra_ligands]

    def code(self):
        """
        Generates a code representation of the ligand.

        Returns:
            The code representation of the ligand.
        """
        return "".join([self._metal.element.name] + self.elements())

    def get_coord(self):
        """
        Returns the coordinates of the metal and ligand atoms.

        Returns:
            An array containing the coordinates of the metal and ligand atoms.
        """
        return np.array([self._metal.pos.tolist()] + [ligand.atom.pos.tolist() for ligand in self._ligands + self._extra_ligands])

    def coordination(self):
        """
        Returns the coordination number of the metal.

        Returns:
            The coordination number.
        """
        return len(self._ligands) + len(self._extra_ligands)

    def contains(self, atom):
        """
        Checks if the ligand contains a specific atom.

        Args:
            atom (gemmi.Atom): The atom to check.

        Returns:
            True if the ligand contains the atom, False otherwise.
        """
        return atom in [ligand.atom for ligand in self._ligands + self._extra_ligands]

    def filter_extra(self):
        """
        Filters out extra ligands that are too close to any ligand atom.
        """
        to_delete = []
        for i, atom1 in enumerate(self._extra_ligands):
            for atom2 in self._ligands:
                if distance(atom1.atom, atom2.atom) < 1:
                    to_delete.append(i)
                    break
        if to_delete:
            self._extra_ligands = [atom for i, atom in enumerate(
                self._extra_ligands) if i not in to_delete]

    def filter_base(self):
        """
        Filters out ligands that are too far from the metal based on covalent distance coefficient.
        """
        to_delete = []
        for i, atom1 in enumerate(self._ligands):
            if self._cov_dist_coeff(atom1) > 2:
                to_delete.append(i)

        if to_delete:
            self._ligands = [atom for i, atom in enumerate(
                self._ligands) if i not in to_delete]

    def mean_occ(self):
        """
        Calculates the mean occupancy of the metal and ligand atoms.

        Returns:
            The mean occupancy.
        """
        return np.mean([self._metal.occ] + [ligand.atom.occ for ligand in self._ligands + self._extra_ligands])

    def mean_b(self):
        """
        Calculates the mean B-factor of the metal and ligand atoms.

        Returns:
            The mean B-factor.
        """
        return np.mean([self._metal.b_iso] + [ligand.atom.b_iso for ligand in self._ligands + self._extra_ligands])

    def __str__(self) -> str:
        return f"{self._metal.name} - {self._chain.name} - {self._residue.name} - {self._residue.seqid.num}"

    def __repr__(self) -> str:
        ligands = " ".join([ligand.atom.name for ligand in self._ligands])
        extra_ligands = " ".join(
            [ligand.atom.name for ligand in self._extra_ligands])
        return f"{self._metal.name} - {self._chain.name} - {self._residue.name} - {self._residue.seqid.num} - {ligands} - {extra_ligands}"


def angle(atom1: gemmi.Atom, atom2: gemmi.Atom, atom3: gemmi.Atom):
    """
    Calculates the angle between three atoms.

    Args:
        atom1 (gemmi.Atom): The first atom.
        atom2 (gemmi.Atom): The second atom.
        atom3 (gemmi.Atom): The third atom.

    Returns:
        The angle in degrees between the three atoms.
    """
    vec1 = atom1.pos - atom2.pos
    vec2 = atom3.pos - atom2.pos
    cosine_angle = vec1.dot(
        vec2) / (np.linalg.norm([vec1.x, vec1.y, vec1.z]) * np.linalg.norm([vec2.x, vec2.y, vec2.z]))
    return np.degrees(np.arccos(cosine_angle))


def distance(atom1, atom2):
    """
    Calculates the Euclidean distance between two atoms.

    Args:
        atom1 (gemmi.Atom): The first atom.
        atom2 (gemmi.Atom): The second atom.

    Returns:
        The Euclidean distance between the two atoms.
    """
    return np.sqrt((atom1.pos - atom2.pos).dot(atom1.pos - atom2.pos))


def get_ligands(st, ligand, bonds=None, max_dist=10, only_best=False) -> list[Ligand]:
    """
    Retrieves ligands associated with a metal in a structure.

    Args:
        st (gemmi.Structure): The structure.
        ligand (str): The name of the ligand.
        bonds (dict): Dictionary of metal-ligand bond information.
        max_dist (float): The maximum distance for ligand searching.
        only_best (bool): Whether to return only the best ligand structures.

    Returns:
        A list of Ligand objects representing the ligands associated with the metal.
    """
    scale = Config().scale()
    alpha = 1.5
    beta1 = [1.2, 1.3, 1.4]
    alpha1 = 1.1
    angle1 = 60

    

    def covalent_radii(element):
        return gemmi.Element(element).covalent_r
    
    def find_min_angle_and_update(atom, n0, n1, beta_c):
        if not n1:
            return n0, n1
        
        while True:
            l_n1 = len(n1)
            angles = [
                (angle(a1.atom, atom, a2.atom), a1, a2)
                for a1 in n1 + n0 for a2 in n1 + n0 if a1 != a2
            ]


            if all(a > angle1 for a, _, _ in angles):
                return n0, n1
            
            angles = list(filter(lambda x: (x[1] in n1) or (x[2] in n1), angles))
            _, min_a1, min_a2 = min(angles, key=lambda x: x[0])

            coef_i = distance(atom, min_a1.atom) / (covalent_radii(atom.element.name) + covalent_radii(min_a1.atom.element.name))
            coef_j = distance(atom, min_a2.atom) / (covalent_radii(atom.element.name) + covalent_radii(min_a2.atom.element.name))
            max_coef_atom = min_a1 if coef_i > coef_j else min_a2

   
            if max_coef_atom in n1 and max(coef_i, coef_j) > beta_c:
                n1.remove(max_coef_atom)
                      
            if len(n1) == l_n1:
                return n0, n1
            
            if not n1:
                return n0, n1

    if st is None:
        return None

    ns = gemmi.NeighborSearch(st[0], st.cell, 5).populate(include_h=False)
    structures = []

    if not bonds:
        bonds = {}

    for chain in st[0]:
        for residue in chain:
            if residue.name != ligand:
                continue

            for atom in residue:
                if atom.element.is_metal:
                    metal_name = atom.name
                    metal_bonds = set(bonds.get(metal_name, []))
                    ligand_obj = Ligand(atom, residue, chain)
                    structures.append(ligand_obj)
                    if Config().simple:
                        marks = ns.find_neighbors(
                            atom, min_dist=0.1, max_dist=max_dist)
                        for mark in marks:
                            cra = mark.to_cra(st[0])
                            if cra.atom.element.is_metal:
                                continue
                            if ligand_obj.contains(cra.atom):
                                continue

                            if bonds:
                                if cra.residue.name == ligand and cra.residue.seqid.num == residue.seqid.num and cra.chain.name == chain.name:
                                    # if cra.residue.name == ligand:
                                    if cra.atom.name in metal_bonds:
                                        ligand_obj.add_ligand(
                                            Atom(cra.atom, cra.residue, cra.chain))
                                elif distance(atom, cra.atom) <= (covalent_radii(atom.element.name) + covalent_radii(cra.atom.element.name)) * scale:
                                    ligand_obj.add_extra_ligand(
                                        Atom(cra.atom, cra.residue, cra.chain))
                            elif distance(atom, cra.atom) <= (covalent_radii(atom.element.name) + covalent_radii(cra.atom.element.name)) * scale:
                                if cra.residue.name == ligand and cra.residue.seqid.num == residue.seqid.num and cra.chain.name == chain.name:
                                    ligand_obj.add_ligand(
                                        Atom(cra.atom, cra.residue, cra.chain))
                                else:
                                    ligand_obj.add_extra_ligand(
                                        Atom(cra.atom, cra.residue, cra.chain))
                    else:
                        k = 2
                        # Step 1: Select all atoms for which d(m, i) < alpha * (r_m + r_i). Denote this set as n1.
                        marks = ns.find_neighbors(atom, min_dist=0.1, max_dist=max_dist)
                        
                        neighbour_atoms = [mark.to_cra(st[0]) for mark in marks]
                        n1 = [
                            neighbour_atom for neighbour_atom in neighbour_atoms
                            if not neighbour_atom.atom.element.is_metal and distance(atom, neighbour_atom.atom) < (covalent_radii(atom.element.name) + covalent_radii(neighbour_atom.atom.element.name)) * alpha
                        ]

                        # Step 2: Select all atoms for which d(m, i) <= alpha1 * (r_m + r_i). Denote this set n0.
                        n0 = [
                            neighbour_atom for neighbour_atom in neighbour_atoms
                            if not neighbour_atom.atom.element.is_metal and distance(atom, neighbour_atom.atom) <= (covalent_radii(atom.element.name) + covalent_radii(neighbour_atom.atom.element.name)) * alpha1
                        ]

                        

                        # Step 3: Remove atoms in n0 from n1n1
                        n1 = [a for a in n1 if a not in n0]
                        if bonds:
                            for a in n1:
                                if a.atom.name in metal_bonds and a.residue.name == ligand and a.residue.seqid.num == residue.seqid.num and a.chain.name == chain.name:
                                    n0.append(a)
                                    n1.remove(a)

                        # Step 4-9: Apply the logic iteratively
                        beta_c = beta1[k]
                        while k >=0 :
                            n0, n1 = find_min_angle_and_update(atom, n0, n1, beta_c)
                            k -= 1
                            beta_c = beta1[k]

                        n0.extend(n1)

                        # Add atoms to ligand_obj
                        for a in n0:
                            if a.residue.name == ligand and a.residue.seqid.num == residue.seqid.num and a.chain.name == chain.name:
                                ligand_obj.add_ligand(Atom(a.atom, a.residue, a.chain))
                            else:
                                ligand_obj.add_extra_ligand(Atom(a.atom, a.residue, a.chain))

                    # ligand_obj.filter_base()
                    ligand_obj.filter_extra()

                    if ligand_obj.ligands_len < len(bonds.get(metal_name, [])):
                        raise ValueError(
                            f"There is inconsistency between ligand(s) in the PDB and monomer file. Metal {metal_name} in {chain.name} - {residue.name} - {residue.seqid.num} has fewer neighbours than expected. Expected: {sorted(bonds.get(metal_name, []))}, found: {sorted([l.atom.name for l in ligand_obj.ligands])}")

    if only_best:
        best_structures = []
        metals = np.unique(
            [structure.metal.name for structure in structures]).tolist()

        for metal_name in metals:
            metal_stuctures = [
                s for s in structures if metal_name == s.metal.name]
            b_vlaues = [metal.mean_b() for metal in metal_stuctures]
            occ_values = [metal.mean_occ() for metal in metal_stuctures]
            best_occ = np.max(occ_values)
            best_structures.append(metal_stuctures[np.argmin(
                np.where(occ_values == best_occ, b_vlaues, np.inf))])
        structures = best_structures

    return structures
