import json
import gemmi


def main(jsonPaths, stPath=None, outputPrefix="restraints", jsonEquivalentsPath=None, keep_links=False):
    #    list       string       string                     string                    bool

    outputRestraintsPath = outputPrefix + ".txt"
    outputRestraintsCootPath = outputPrefix + "_coot.txt"
    outputRestraintsPhenixPath = outputPrefix + ".params"
    outputMmcifPath = outputPrefix + ".mmcif"
    outputPdbPath = outputPrefix + ".pdb"
    
    if stPath:
        st = gemmi.read_structure(stPath)
        if not keep_links:
            # Delete LINK connection records of a type of MetalC or involving metal atoms
            connections_kept = gemmi.ConnectionList()
            for connection in st.connections:
                for m in range(len(st)):
                    partner1_atom = st[m].find_cra(connection.partner1, ignore_segment=True).atom
                    partner2_atom = st[m].find_cra(connection.partner2, ignore_segment=True).atom
                    if partner1_atom and partner2_atom:  # check if atoms exist
                        if not (partner1_atom.element.is_metal or \
                                partner2_atom.element.is_metal):
                            connections_kept.append(connection)
            n_links_old = len(st.connections)
            n_links_kept = len(connections_kept)
            st.connections.clear()
            st.connections = connections_kept
            n_links_deleted = n_links_old - n_links_kept
            print(f"Number of deleted links related to metals from the input structure file: {n_links_deleted}")
    
    # Load JSON files from metalCoord
    d = []
    for jsonPath in jsonPaths:
        with open(jsonPath, "r") as f:
            d.extend(json.load(f))
    
    # If provided, load a JSON file describing equivalent atoms (mainly for metal-atom-atom-metal triangles)
    # and apply to the JSON information from metalCoord
    # - modify records in 'base' and 'pdb' section for exte bond restraints
    # - delete records in 'angles' section for exte angle restraints
    if jsonEquivalentsPath:
        print("JSON file describing equivalent atoms will be applied.")
        atom_pairs_all = []
        with open(jsonEquivalentsPath, "r") as f:
            atom_pairs_all.extend(json.load(f))
        for atom_pairs in atom_pairs_all:
            for m, atom_metal in enumerate(d):
                for l, ligand in enumerate(atom_metal['ligands']):
                    for base_or_pdb in ["base", "pdb"]:
                        for j, atom_ligand in enumerate(ligand[base_or_pdb]):
                                if atom_pairs[0] == d[m]['ligands'][l][base_or_pdb][j]['ligand']:
                                    print("Bond length: Template atom:", atom_pairs[0])
                                    d[m]['ligands'][l][base_or_pdb].append(d[m]['ligands'][l][base_or_pdb][j].copy())
                                    d[m]['ligands'][l][base_or_pdb][-1]['ligand'] = atom_pairs[1]
                                    print("Bond length: Added atom:   ", atom_pairs[1])
                    angle_restraints_to_delete = []
                    for j, atom_ligands in enumerate(ligand['angles']):
                        for ligand in ["ligand1", "ligand2"]:
                            if atom_pairs[0] == d[m]['ligands'][l]['angles'][j][ligand]:
                                angle_restraints_to_delete.append(j)
                                print("Bond angle:  Deleted atom: ", atom_pairs[0])
                    if angle_restraints_to_delete:
                        d[m]['ligands'][l]['angles'] = \
                            [i for j, i in enumerate(d[m]['ligands'][l]['angles']) if j not in angle_restraints_to_delete]
    outputLines = []
    outputLinesCoot = []
    outputLinesPhenix = []
    i_con = 1
    outputLines.append("### RESTRAINTS FROM METALCOORD - BEGINNING ###\n")
    outputLines.append("# Note that this file is not compatible with Coot \n")
    outputLinesPhenix.append("refinement.geometry_restraints.edits {\n")
    for atom_metal in d:  # for every metal atom
        if not atom_metal['icode']: atom_metal['icode'] = "."
        if atom_metal['icode'] != ".":
            atom_metal['sequence_icode'] = str(atom_metal['sequence']) + str(atom_metal['icode'])
        else:
            atom_metal['sequence_icode'] = str(atom_metal['sequence'])
        for l, ligand in enumerate(atom_metal['ligands']):
            if l >= 1: break  # consider only the best matching class - which is listed at the first one 
            outputLines.append(f"# {atom_metal['chain']}{atom_metal['sequence_icode']}/{atom_metal['metal']} {atom_metal['altloc']} {atom_metal['residue']}; class: {atom_metal['ligands'][l]['class']}; procrustes: {str(atom_metal['ligands'][l]['procrustes'])}; coordination: {str(atom_metal['ligands'][l]['coordination'])}; count: {str(atom_metal['ligands'][l]['count'])}; description: {str(atom_metal['ligands'][l]['description'])}\n")
            # exte dist - 'base' for within ligand, 'pdb' for its neighbourhood
            for j, atom_ligand in enumerate(ligand['base'] + ligand['pdb']):
                if not atom_ligand['ligand']['icode']: atom_ligand['ligand']['icode'] = "."
                if atom_ligand['ligand']['icode'] != ".":
                    atom_ligand['ligand']['sequence_icode'] = str(atom_ligand['ligand']['sequence']) + str(atom_ligand['ligand']['icode'])
                else:
                    atom_ligand['ligand']['sequence_icode'] = str(atom_ligand['ligand']['sequence'])
                for i in range(len(atom_ligand['distance'])):
                    line = f"exte dist first chain {atom_metal['chain']} resi {atom_metal['sequence']} inse {atom_metal['icode']} atom {atom_metal['metal']} "
                    line_coot = line
                    atom_selection_1_phenix = f"chain {atom_metal['chain']} and resname {atom_metal['residue']} and resid {atom_metal['sequence_icode']} and name {atom_metal['metal']}"
                    if atom_metal['altloc']: # not for Coot
                        line += f"altecode {atom_metal['altloc']} "
                        atom_selection_1_phenix += f" and altloc {atom_metal['altloc']}"
                    line += f"second chain {atom_ligand['ligand']['chain']} resi {atom_ligand['ligand']['sequence']} inse {atom_ligand['ligand']['icode']} atom {atom_ligand['ligand']['name']} "
                    line_coot += f"second chain {atom_ligand['ligand']['chain']} resi {atom_ligand['ligand']['sequence']} inse {atom_ligand['ligand']['icode']} atom {atom_ligand['ligand']['name']} "
                    atom_selection_2_phenix = f"chain {atom_ligand['ligand']['chain']} and resname {atom_ligand['ligand']['residue']} and resid {atom_ligand['ligand']['sequence_icode']} and name {atom_ligand['ligand']['name']}"
                    if atom_ligand['ligand']['altloc']: # not for Coot
                        line += f"altecode {atom_ligand['ligand']['altloc']} "
                        atom_selection_2_phenix += f" and altloc {atom_ligand['ligand']['altloc']}"
                    line += f"value {atom_ligand['distance'][i]} sigma {atom_ligand['std'][i]}"
                    line_coot += f"value {atom_ligand['distance'][i]} sigma {atom_ligand['std'][i]}"
                    if atom_ligand['ligand']['symmetry']:
                        line += " symm y"
                    if i == 0:
                        line += f" type 0"
                    else:
                        line += f" type 1"
                    if len(atom_ligand['distance']) >= 2:
                        line += f" # variant {str(i + 1)}"
                    # print(line)
                    outputLines.append(line + "\n")
                    # restaints for Coot cannot include atoms with altloc and symmetry identifiers
                    if not atom_metal['altloc'] and \
                            not atom_ligand['ligand']['altloc'] and \
                            not atom_ligand['ligand']['symmetry']:
                        outputLinesCoot.append(line_coot + "\n")
                    # restaints for phenix.refine: atoms with symmetry records are not included
                    if not atom_ligand['ligand']['symmetry']:
                        outputLinesPhenix.append("  bond {\n")
                        outputLinesPhenix.append(f"    action = *add\n")
                        outputLinesPhenix.append(f"    atom_selection_1 = {atom_selection_1_phenix}\n")
                        outputLinesPhenix.append(f"    atom_selection_2 = {atom_selection_2_phenix}\n")
                        outputLinesPhenix.append(f"    distance_ideal = {atom_ligand['distance'][i]}\n")
                        outputLinesPhenix.append(f"    sigma = {atom_ligand['std'][i]}\n")
                        outputLinesPhenix.append("  }\n")
                    if j >= len(ligand['base']) and stPath:
                        # create a mmCIF link, it's from 'pdb', i.e. neighbourhood
                        con = gemmi.Connection()
                        con.name = 'metalcoord' + str(i_con)
                        con.type = gemmi.ConnectionType.MetalC
                        if atom_metal['altloc']:
                            atom_metal_altloc = atom_metal['altloc']
                        else:
                            atom_metal_altloc = '*' # take the first matching atom regardless of altloc
                        con.partner1 = gemmi.make_address(
                            st[0][atom_metal['chain']],
                            st[0][atom_metal['chain']][atom_metal['sequence_icode']][atom_metal['residue']],
                            st[0][atom_metal['chain']][atom_metal['sequence_icode']][atom_metal['residue']].find_atom(atom_metal['metal'], atom_metal_altloc))
                        if atom_ligand['ligand']['altloc']:
                            atom_ligand_altloc = atom_ligand['ligand']['altloc']
                        else:
                            atom_ligand_altloc = '*' # take the first matching atom regardless of altloc
                        con.partner2 = gemmi.make_address(
                            st[0][atom_ligand['ligand']['chain']],
                            st[0][atom_ligand['ligand']['chain']][atom_ligand['ligand']['sequence_icode']][atom_ligand['ligand']['residue']],
                            st[0][atom_ligand['ligand']['chain']][atom_ligand['ligand']['sequence_icode']][atom_ligand['ligand']['residue']].find_atom(atom_ligand['ligand']['name'], atom_ligand_altloc))
                        # con.reported_distance = atom_ligand['distance'][i]
                        con_exists = False
                        for con_existing in st.connections:
                            if (con_existing.partner1 == con.partner1 and con_existing.partner2 == con.partner2) or \
                                    (con_existing.partner1 == con.partner2 and con_existing.partner2 == con.partner1):
                                con_exists = True
                        if not con_exists:
                            st.connections.append(con)
                            i_con += 1
            # exte angle
            for atom_ligands in ligand['angles']:  # if any angles specified in ligand['angles']
                if not atom_ligands['ligand1']['icode']: atom_ligands['ligand1']['icode'] = "."
                if not atom_ligands['ligand2']['icode']: atom_ligands['ligand2']['icode'] = "."
                if atom_ligands['ligand1']['icode'] != ".":
                    atom_ligands['ligand1']['sequence_icode'] = str(atom_ligands['ligand1']['sequence']) + str(atom_ligands['ligand1']['icode'])
                else:
                    atom_ligands['ligand1']['sequence_icode'] = str(atom_ligands['ligand1']['sequence'])
                if atom_ligands['ligand2']['icode'] != ".":
                    atom_ligands['ligand2']['sequence_icode'] = str(atom_ligands['ligand2']['sequence']) + str(atom_ligands['ligand2']['icode'])
                else:
                    atom_ligands['ligand2']['sequence_icode'] = str(atom_ligands['ligand2']['sequence'])
                line = f"exte angle "
                # ligand1
                line += f"first chain {atom_ligands['ligand1']['chain']} resi {atom_ligands['ligand1']['sequence']} inse {atom_ligands['ligand1']['icode']} atom {atom_ligands['ligand1']['name']} "
                line_coot = line
                atom_selection_1_phenix = f"chain {atom_ligands['ligand1']['chain']} and resname {atom_ligands['ligand1']['residue']} and resid {atom_ligands['ligand1']['sequence_icode']} and name {atom_ligands['ligand1']['name']}"
                if atom_ligands['ligand1']['altloc']:
                    line += f"altecode {atom_ligands['ligand1']['altloc']} "
                    atom_selection_1_phenix += f" and altloc {atom_ligands['ligand1']['altloc']}"
                if atom_ligands['ligand1']['symmetry']:
                    line += "symm y "
                # metal
                line += f"next chain {atom_metal['chain']} resi {atom_metal['sequence']} inse {atom_metal['icode']} atom {atom_metal['metal']} "
                line_coot += f"next chain {atom_metal['chain']} resi {atom_metal['sequence']} inse {atom_metal['icode']} atom {atom_metal['metal']} "
                atom_selection_2_phenix = f"chain {atom_metal['chain']} and resname {atom_metal['residue']} and resid {atom_metal['sequence_icode']} and name {atom_metal['metal']}"
                if atom_metal['altloc']:
                    line += f"altecode {atom_metal['altloc']} "
                    atom_selection_2_phenix += f" and altloc {atom_metal['altloc']}"
                # ligand2
                line += f"next chain {atom_ligands['ligand2']['chain']} resi {atom_ligands['ligand2']['sequence']} inse {atom_ligands['ligand2']['icode']} atom {atom_ligands['ligand2']['name']} "
                line_coot += f"next chain {atom_ligands['ligand2']['chain']} resi {atom_ligands['ligand2']['sequence']} inse {atom_ligands['ligand2']['icode']} atom {atom_ligands['ligand2']['name']} "
                atom_selection_3_phenix = f"chain {atom_ligands['ligand2']['chain']} and resname {atom_ligands['ligand2']['residue']} and resid {atom_ligands['ligand2']['sequence_icode']} and name {atom_ligands['ligand2']['name']}"
                if atom_ligands['ligand2']['altloc']:
                    line += f"altecode {atom_ligands['ligand2']['altloc']} "
                    atom_selection_3_phenix += f" and altloc {atom_ligands['ligand2']['altloc']}"
                if atom_ligands['ligand2']['symmetry']:
                    line += "symm y "
                line += f"value {round(atom_ligands['angle'], 2)} sigma {round(atom_ligands['std'], 2)}"
                line_coot += f"value {round(atom_ligands['angle'], 2)} sigma {round(atom_ligands['std'], 2)}"
                line += f" type 0"
                # print(line)
                outputLines.append(line + "\n")
                # restaints for Coot cannot include atoms with altloc and symmetry identifiers
                if not atom_metal['altloc'] and \
                        not atom_ligands['ligand1']['altloc'] and \
                        not atom_ligands['ligand1']['symmetry'] and \
                        not atom_ligands['ligand2']['altloc'] and \
                        not atom_ligands['ligand2']['symmetry']:
                    outputLinesCoot.append(line_coot + "\n")
                # restaints for phenix.refine cannot include atoms with symmetry
                if not atom_ligands['ligand1']['symmetry'] and \
                        not atom_ligands['ligand2']['symmetry']:
                    outputLinesPhenix.append("  angle {\n")
                    outputLinesPhenix.append(f"    action = *add\n")
                    outputLinesPhenix.append(f"    atom_selection_1 = {atom_selection_1_phenix}\n")
                    outputLinesPhenix.append(f"    atom_selection_2 = {atom_selection_2_phenix}\n")
                    outputLinesPhenix.append(f"    atom_selection_3 = {atom_selection_3_phenix}\n")
                    outputLinesPhenix.append(f"    angle_ideal = {round(atom_ligands['angle'], 2)}\n")
                    outputLinesPhenix.append(f"    sigma = {round(atom_ligands['std'], 2)}\n")
                    outputLinesPhenix.append("  }\n")
    outputLines.append("### RESTRAINTS FROM METALCOORD - END ###\n")
    outputLinesPhenix.append("}\n")
    with open(outputRestraintsPath, "w") as f:
        f.writelines(outputLines)
    with open(outputRestraintsCootPath, "w") as f:
        f.writelines(outputLinesCoot)
    with open(outputRestraintsPhenixPath, "w") as f:
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
        description="Convert JSON files from metalCoord to external restraint keywords for servalcat / refmac5 / coot / phenix.refine"
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
        help="PDB or mmCIF file whose LINK records will be updated",
        type=str,
        metavar="STRUCTURE",
    )
    parser.add_argument(
        "-o",
        help="Prefix for output file names",
        type=str,
        default="output",
        metavar="OUTPUT"
    )
    parser.add_argument(
        "--keep-links",
        help="Keep link connection records related to metals in the input structure file",
        action="store_true",
    )
    parser.add_argument(
        "-e",
        help="JSON file describing equivalent atoms",
        type=str,
        metavar="JSON_EQUIVALENTS",
    )

    args = parser.parse_args()
    if len(args.i) < 2:
        jsonPaths = [args.i]
    else:
        jsonPaths = args.i
    stPath = args.p
    outputPrefix = args.o
    jsonEquivalentsPath = args.e
    keep_links = args.keep_links

    main(jsonPaths, stPath, outputPrefix, jsonEquivalentsPath, keep_links)
