from pathlib import Path
import tempfile
import argparse
import sys
import logging
import pandas as pd
import functools

from Bio.PDB import PDBParser, PDBIO, Select, NeighborSearch
from Bio.PDB.SASA import ShrakeRupley
from Bio.SeqUtils import seq1
from Bio.SeqUtils.ProtParam import ProteinAnalysis

import multiprocess as mp  # pip install multiprocess

# ======================
# Module overview
# ======================

# This script summarizes antibody-antigen complexes from SAbDab.
# For each complex the script: (1) reads a PDB file, (2) computes per-residue
# solvent accessible surface area (SASA) for the complex and for the antigen
# alone, (3) identifies antigen residues whose SASA decreases on binding
# (ΔSASA) and are within a distance cutoff to binder atoms — these are
# considered epitope residues, and (4) exports a summary CSV with sequences
# and properties useful for downstream analysis.


# ======================
# Config & data loading (defaults)
# ======================

# These are module-level defaults and can be overridden via CLI flags.
# - SABDAB_PARQUET: path to a parquet file exported from SAbDab that contains
#   a column `complex_fname` listing PDB filenames for each complex.
SABDAB_PARQUET = Path(r"/homefs/home/boddyc/sabdab_2025-05-14.parquet")

# - PDB_DIR: default directory where PDB files are stored (one file per complex)
PDB_DIR = Path("./sabdab")

# Distance cutoff (in Å) for considering an antigen residue as being in
# contact with binder atoms (non-hydrogen atoms on antibodies).
DIST_CUTOFF = 4.5

# ΔSASA threshold (Å^2). If the residue loses at least this much SASA on
# complex formation and is in contact, we mark it as part of the epitope.
DSASA_THRESH = 2.0

# pH used when computing residue charge properties (via Biopython ProtParam)
PH = 7.4

# Antigen and antibody chain defaults. These can be changed if a dataset
# uses different conventions.
ANTIGEN_CHAINS = ["A"]
LIGHT_CHAIN = "L"
HEAVY_CHAIN = "H"


# ======================
# Helpers
# ======================

class OnlyChains(Select):
    """Select only a set of chains when writing a PDB with PDBIO.

    Biopython's PDBIO takes a Select subclass to filter which chains/residues
    to write. We use this when creating an antigen-only PDB from the full
    complex.
    """
    def __init__(self, keep):
        # keep: iterable of chain IDs (e.g. ['A'])
        self.keep = set(keep)

    def accept_chain(self, chain):
        # Return True if the chain ID should be kept
        return chain.id in self.keep

def per_residue_sasa(model):
    """Compute per-residue solvent accessible surface area (SASA).

    Returns a dictionary keyed by (chain_id, residue_number, insertion_code)
    with the total SASA for that residue (sum of atom SASAs). This uses
    Biopython's Shrake-Rupley implementation which annotates each atom with a
    ``sasa`` attribute after computing.
    """
    sr = ShrakeRupley()
    # compute SASA at the atomic level; 'level="A"' leaves results on atoms
    sr.compute(model, level="A")

    res_sasa = {}
    # iterate chains and residues in the model
    for chain in model:
        for res in chain:
            # skip hetero/water residues: Biopython encodes these with a non-space
            # in res.id[0]
            if res.id[0] != " ":
                continue
            # sum per-atom sasa values (some atoms may not have 'sasa' attribute,
            # so use getattr with a default)
            sasa = sum(getattr(atom, "sasa", 0.0) for atom in res)
            # Use the tuple (chain_id, residue_number, insertion_code) as key
            res_sasa[(chain.id, res.id[1], (res.id[2] or "").strip())] = float(sasa)
    return res_sasa

def chain_seq(model, chain_id):
    """Return the amino-acid one-letter sequence for a chain in the model.

    Notes:
    - Uses Biopython's seq1 to convert three-letter residue names to one-letter
      codes. Special residues like selenocysteine (SEC) are mapped to 'U'.
    - Skips hetero/water residues.
    """
    seq_chars = []
    for ch in model:
        if ch.id != chain_id:
            continue
        for res in ch:
            if res.id[0] != " ":
                continue
            aa = seq1(res.get_resname(), custom_map={"SEC": "U", "PYL": "O"})
            if aa and len(aa) == 1:
                seq_chars.append(aa)
    return "".join(seq_chars)

def residue_in_contact(res, ns, cutoff=DIST_CUTOFF):
    """Return True if any non-hydrogen atom of `res` is within `cutoff` Å
    of any atom in the NeighborSearch `ns`.

    NeighborSearch provides a fast spatial index for atom coordinates. We
    ignore hydrogen atoms (commonly missing or inconsistently named in PDBs).
    """
    for atom in res:
        if atom.element == "H":
            continue
        # ns.search returns a list of atoms within cutoff; if non-empty -> contact
        if ns.search(atom.coord, cutoff):
            return True
    return False

def aa_props(aa1, ph):
    """Compute simple amino-acid properties for a single-residue string.

    Returns a tuple of (hydrophobicity, charge_at_pH). We use Biopython's
    ProteinAnalysis which expects a peptide sequence; for a single residue this
    still works. If the calculation fails we return (None, None).
    """
    try:
        pa = ProteinAnalysis(aa1)
        return pa.gravy(), pa.charge_at_pH(ph)
    except Exception:
        return None, None

def antigen_seq_and_props(model, ag_chains, ph):
    """Return the concatenated antigen sequence and per-residue properties.

    This walks through the specified antigen chains in `ag_chains` and
    collects:
    - the full antigen sequence (string)
    - a list of hydrophobicity values (one per residue)
    - a list of charges at the requested pH (one per residue)

    These property lists are aligned to the antigen sequence and can be used
    to analyze epitope physicochemical patterns.
    """
    seq_chars, hydros, charges = [], [], []
    for ch in model:
        if ch.id not in ag_chains:
            continue
        for res in ch:
            if res.id[0] != " ":
                continue
            aa = seq1(res.get_resname(), custom_map={"SEC": "U", "PYL": "O"})
            if not (aa and len(aa) == 1):
                continue
            seq_chars.append(aa)
            h, c = aa_props(aa, ph)
            hydros.append(h)
            charges.append(c)
    return "".join(seq_chars), hydros, charges


# ======================
# Worker
# ======================

def process_one(pdb_id, pdb_dir: Path = None):
    """Process one complex and return a summary dictionary.

    This function is intended to run inside a worker process. It performs the
    following steps:
    1. Locate the PDB file (in `pdb_dir` or the default `PDB_DIR`).
    2. Parse the structure and extract model 0 (first model in the file).
    3. Create an antigen-only temporary PDB and parse it to compute "free"
       SASA for the antigen (i.e. as if it were unbound).
    4. Compute atomic SASA for complex and free antigen, then per-residue SASA.
    5. Use a spatial NeighborSearch on binder atoms to detect contact residues.
    6. For antigen residues that meet the contact and ΔSASA thresholds,
       record the residue as part of the epitope.

    Returns: dict with keys like 'PDBID', 'antigen_seq', 'epitope_positions',
    and per-residue property lists. On failure returns None so the caller can
    skip problematic entries.
    """
    try:
        # Initialize a PDB parser - create inside worker to avoid pickling issues
        parser = PDBParser(QUIET=True)

        # Allow overriding the PDB directory via the function argument
        base_dir = pdb_dir or PDB_DIR
        pdb_path = base_dir / pdb_id

        if not pdb_path.exists():
            # Caller will see the warning and skip this PDB
            raise FileNotFoundError(f"PDB file not found: {pdb_path}")

        # Parse the PDB file and take the first model (model[0])
        structure = parser.get_structure(pdb_id, pdb_path)
        model = structure[0]

        # Write the antigen-only coordinates to a temporary PDB file so we
        # can compute the "free" SASA. Using a temporary file avoids keeping
        # large structures in memory or colliding filenames between workers.
        with tempfile.TemporaryDirectory() as td:
            antigen_path = Path(td) / f"{pdb_id}_antigen_only.pdb"
            pdbio = PDBIO()
            pdbio.set_structure(structure)
            # Select only antigen chains and write a new PDB
            pdbio.save(str(antigen_path), select=OnlyChains(ANTIGEN_CHAINS))

            # Parse that antigen-only PDB to compute SASA on the unbound antigen
            antigen_model = parser.get_structure(f"{pdb_id}_ag", str(antigen_path))[0]

        # Compute per-residue SASA for complex and antigen-alone
        sasa_complex = per_residue_sasa(model)
        sasa_free = per_residue_sasa(antigen_model)
        antigen_delta_sasa = []

        # Build a list of binder atoms (all atoms in non-antigen chains, excluding H)
        binder_atoms = []
        for ch in model:
            if ch.id in ANTIGEN_CHAINS:
                continue
            for res in ch:
                if res.id[0] != " ":
                    continue
                for atom in res:
                    if atom.element != "H":
                        binder_atoms.append(atom)

        # Create a spatial index for fast neighbour queries
        ns_binder = NeighborSearch(binder_atoms)

        # Now evaluate each antigen residue for contact and ΔSASA
        residues = []
        for ch in model:
            if ch.id not in ANTIGEN_CHAINS:
                continue
            for res in ch:
                if res.id[0] != " ":
                    continue
                rid = (ch.id, res.id[1], (res.id[2] or "").strip())
                d_asa = sasa_free.get(rid, 0.0) - sasa_complex.get(rid, 0.0)
                contact = residue_in_contact(res, ns_binder, DIST_CUTOFF)
                residues.append((rid[1], res.get_resname(), contact and (d_asa >= DSASA_THRESH)))
                antigen_delta_sasa.append(float(d_asa))

        # Extract epitope positions and residue types
        epi_positions = [r[0] for r in residues if r[2]]
        epi_resnames = [r[1] for r in residues if r[2]]
        epi_residues = [seq1(rn, custom_map={"SEC": "U", "PYL": "O"}) for rn in epi_resnames]

        # Get the antigen sequence and per-residue properties (hydrophobicity, charge)
        antigen_seq_str, antigen_hydros, antigen_charges = antigen_seq_and_props(model, ANTIGEN_CHAINS, PH)

        # Build and return the summary row
        return {
            "PDBID": pdb_id,
            "antigen_seq": "".join(chain_seq(model, ch) for ch in ANTIGEN_CHAINS),
            "light_chain_seq": chain_seq(model, LIGHT_CHAIN),
            "heavy_chain_seq": chain_seq(model, HEAVY_CHAIN),
            "epitope_positions": epi_positions,
            "epitope_residues": epi_residues,
            "antigen_delta_sasa": antigen_delta_sasa,
            "antigen_hydrophobicity": antigen_hydros,
            "antigen_charge": antigen_charges,
        }

    except Exception as e:
        # If anything goes wrong we log/print a warning and return None so the
        # main loop can skip this entry without crashing the whole job.
        print(f"[WARN] {pdb_id}: {e}")
        return None


# ======================
# Main CLI entrypoint
# ======================

def main():
    """CLI entrypoint.

    Parses command-line arguments, loads the SAbDab parquet to obtain the list
    of PDB filenames, runs the worker pool, performs a small QC check, and
    writes the results to CSV.
    """

    parser = argparse.ArgumentParser(
        description="Extract epitope summaries from SAbDab complexes (local PDB files expected)."
    )
    parser.add_argument("--sabdab", type=Path, default=SABDAB_PARQUET, help="Path to SAbDab parquet file")
    parser.add_argument("--pdb_dir", type=Path, default=PDB_DIR, help="Directory with PDB files named by complex_fname")
    parser.add_argument("--out", type=Path, default=Path("complex_summaries.csv"), help="Output CSV path")
    parser.add_argument("--workers", type=int, default=mp.cpu_count(), help="Number of worker processes")
    args = parser.parse_args()

    # Simple logging configuration — use DEBUG/INFO depending on needs
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    # Validate inputs early
    if not args.sabdab.exists():
        logging.error("SAbDab parquet not found: %s", args.sabdab)
        sys.exit(2)

    # Read the parquet and extract unique PDB filenames
    sabdab = pd.read_parquet(args.sabdab)
    PDB_IDS = (
        sabdab["complex_fname"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    logging.info("Loaded %d PDB IDs from SAbDab.", len(PDB_IDS))

    # Choose a chunksize. This balances the scheduling overhead vs. load balancing
    chunksize = max(1, len(PDB_IDS) // (args.workers * 8) or 1)

    # Run workers in parallel. We bind the user-specified pdb_dir into the
    # worker function using functools.partial so Pool.map receives a single-arg
    # callable (pdb_id -> result).
    rows = []
    worker_fn = functools.partial(process_one, pdb_dir=args.pdb_dir)
    with mp.Pool(processes=args.workers, maxtasksperchild=10) as pool:
        for row in pool.imap_unordered(worker_fn, PDB_IDS, chunksize=chunksize):
            if row is not None:
                rows.append(row)

    # Build final dataframe and perform quick consistency checks
    df = pd.DataFrame(rows)
    logging.info("Built DataFrame with %d rows.", len(df))

    # Quality control: verify epitope/residue property list lengths match
    epi_ok = df.apply(
        lambda r: len(r["epitope_positions"]) == len(r["epitope_residues"]),
        axis=1,
    )
    props_ok = df.apply(
        lambda r: (
            len(r["antigen_seq"]) == len(r["antigen_delta_sasa"]) == len(r["antigen_hydrophobicity"]) == len(r["antigen_charge"]) 
        ),
        axis=1,
    )

    logging.info("=== Quality Control ===")
    logging.info("Total rows                         : %d", len(df))
    logging.info("Epitope length matches             : %d / %d", epi_ok.sum(), len(epi_ok))
    logging.info("Antigen property length matches    : %d / %d", props_ok.sum(), len(props_ok))

    # Export to CSV
    df.to_csv(args.out, index=False)
    logging.info("Saved %s", args.out)

if __name__ == "__main__":
    main()