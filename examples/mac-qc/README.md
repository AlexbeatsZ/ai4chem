# HCN/HNC capability example

The three-atom geometry is from geomeTRIC's upstream HCN/HNC IRC example,
retrieved on 2026-11-23:

https://github.com/leeping/geomeTRIC/tree/master/examples/1-simple-examples/hcn_hnc_irc/qchem

The upstream input uses neutral singlet HF/3-21G. This project preserves those
settings but supplies the energies, analytic gradients and Hessians with PySCF.
`scripts/check_mac_ts_irc.py` additionally optimizes the TS, checks its single
imaginary frequency, runs both IRC directions in separate processes, optimizes
both endpoints and checks their stable frequencies and HCN/HNC connectivity.

This is an independent software capability test, not a new v0.5.0 reaction or
Gaussian validation result. Its runtime does not estimate the cost of the
ten-atom wB97X/6-31G(d) production reaction chains.
