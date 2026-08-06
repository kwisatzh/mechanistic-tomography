# Paper source

`source/nt_mi_control_position_v21-v2.tex` is the public version 2 manuscript.
It builds with a current TeX Live distribution and the `acmart` class:

```bash
cd paper/source
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  nt_mi_control_position_v21-v2.tex
```

The compiled PDF is also available at
`assets/mechanistic-tomography-v2.pdf` from the repository root.

No venue-specific private working copy is included in this repository.
