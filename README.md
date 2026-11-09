# GROVE: Functional-Group Tokenizer for Polymers

**GROVE** (functional-**GR**oup **O**rdered **V**isual **E**ncoding) converts a polymer SMILES string into a sequence of functional-group tokens using 64 hand-designed SMARTS patterns.

Each atom in the SMILES is assigned to exactly one functional group via a greedy maximum-match priority rule, producing a human-interpretable token sequence that preserves the chemical structure of the repeat unit.

GROVE is used as the input representation for the [Motiva](https://github.com/open-matter/Motiva) polymer language model.

---

## Installation

```bash
pip install rdkit numpy pandas scikit-learn
```

---

## Quick Start

```python
from grove import smiles_to_grove

# Polyimide repeat unit
smiles = "*c1ccc(Oc2ccc(-c3ccc(Oc4ccc(N5C(=O)c6ccccc6C5=O)cc4)cc3)cc2)cc1"
grove_string = smiles_to_grove(smiles)
print(grove_string)
```

Command-line usage:

```bash
python grove.py "*OC(C)C(=O)*"
```

---

## Repository Contents

| File | Description |
|------|-------------|
| `grove.py` | GROVE encoder: SMILES → token sequence (1971 lines) |
| `element_to_AtomicNumber.py` | Atomic symbol → atomic number mapping |
| `FunGroup_token.csv` | 64 SMARTS functional-group pattern definitions |
| `tokenizer/PolymerSumTokenization.py` | HuggingFace-compatible tokenizer wrapping GROVE vocabulary |
| `tokenizer/tokenizer_generate.py` | Script to regenerate the vocabulary from scratch |

---

## The 64 SMARTS Patterns

`FunGroup_token.csv` defines 64 functional groups covering:
- Aromatic rings (benzene, naphthalene, thiophene, furan, pyridine, …)
- Aliphatic rings (cyclopropane through cycloheptane)
- Carbonyl-containing groups (ketone, ester, amide, imide, anhydride, …)
- Heteroatom groups (ether, sulfide, sulfone, phosphonate, siloxane, …)
- Halogen substituents (F, Cl, Br, I)
- Backbone atoms (C, N, O, S, …)

---

## Citation

If you use GROVE in your research, please cite:

```
@article{grove2026,
  title   = {GROVE: A Universal Functional-Group Language for Data-Efficient Molecular Property Prediction},
  author  = {Laplacian},
  journal = {Nature Machine Intelligence},
  year    = {2026},
  note    = {Under review}
}
```
