# Proposal checks

These are the standalone scripts that produced the numbers in the written proposal. Each script is one
file and needs only NumPy. The package in `gelucheck/` and the scripts in `scripts/` replace them. Use those
for new work and for the final report.

| Script | What it checks | Replaced by |
|---|---|---|
| `check_section4.py` | Route B: error bound, table size, certification rate, checksum mod 65537 | `scripts/run_route_b.py` |
| `check_routeA.py` | Route A: false alarms and detection for each tolerance | `scripts/run_route_a.py` |
| `check_new_proposal.py` | Formulas in the proposal (tanh/sigmoid forms, batch identity, tail bound) | `scripts/check_proposal_math.py` |

Run one with `.venv/bin/python proposal/check_section4.py`.
