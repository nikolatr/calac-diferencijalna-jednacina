# Diferencijalna jednačina - Masa-opruga-prigušivač sistem

Numeričko rešavanje diferencijalne jednačine oscilatornog sistema sa prinudnim oscilacijama.

## Opis problema

Projekat rešava diferencijalnu jednačinu za sistem masa-opruga-prigušivač pod dejstvom spoljašnje periodične sile:

```
M x'' + μ x' + c x = F = A P sin(2πt)
```

gde su:
- `M` - masa (kg)
- `μ` - koeficijent prigušenja
- `c` - konstanta opruge (N/m)
- `A` - površina (m²)
- `P` - pritisak (N/m²)
- `x` - pomeranje
- `t` - vreme

## Korišćenje

Otvori `diferencijalna-jednacina.ipynb` u Jupyter notebook-u i pokreni sve ćelije.

## Rezultati

Notebook generiše:
1. **Vremenska domena** - Grafik pomeranja `x(t)` i brzine `v(t)` tokom vremena
2. **Analiza prolaznog režima** - Detaljan prikaz početnih oscilacija
3. **Ustaljeni režim** - Ponašanje sistema u ravnotežnom stanju
4. **Frekvencijska analiza** - Spektar snage brzine u ustaljenom režimu

## Zavisnosti

- Python ≥ 3.9
- NumPy - Numerička matematika
- SciPy - Rešavanje diferencijalnih jednačina
- Matplotlib - Vizualizacija rezultata
- ipykernel - Jupyter kernel podrška

## Licenca

MIT
