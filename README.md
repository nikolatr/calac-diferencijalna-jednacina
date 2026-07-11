# Diferencijalna jednačina — sistem masa-opruga-prigušivač

Numerička simulacija prinudno pobuđenog linearnog oscilatora:

$$M\ddot{x} + \mu\dot{x} + cx = F(t), \qquad
F(t) = AP\sin(2\pi f t).$$

Podrazumevana pobuda ima frekvenciju $f=1\ \mathrm{Hz}$ i amplitudu
$AP=200\ \mathrm{N}$.

| Simbol | Opis | Podrazumevana vrednost | Jedinica |
|---|---|---:|---|
| $M$ | masa | 0,032 | kg |
| $\mu$ | koeficijent viskoznog prigušenja | 1 | N·s/m |
| $c$ | krutost opruge | 300 000 | N/m |
| $A$ | površina | 0,00005 | m² |
| $P$ | pritisak | 4 000 000 | N/m² |
| $f$ | frekvencija pobude | 1 | Hz |

Napomena: $\mu$ je dimenzioni koeficijent prigušenja, ne bezdimenzioni faktor
prigušenja $\zeta$. Za zadate vrednosti važi $\zeta\approx0{,}0051$.

## Brzi početak

Potreban je Python 3.9 ili noviji i [uv](https://docs.astral.sh/uv/).

```bash
uv sync --locked
uv run --locked jupyter notebook diferencijalna-jednacina.ipynb
```

Notebook prikazuje pomeranje i brzinu, prolazni i ustaljeni režim, kao i
amplitudski spektar brzine.

## Animacija

![Animacija sistema](animacija.gif)

Postojeća animacija se reprodukuje u realnom vremenu i može se ponovo
generisati ovako:

```bash
uv run --locked python animacija.py
```

Izlazna putanja, broj frejmova u sekundi, brzina reprodukcije i rezolucija su
podesivi:

```bash
uv run --locked python animacija.py \
  --output rezultat.gif --fps 25 --brzina 0.5 --dpi 80
```

`--brzina 0.5` daje dvostruko sporiju reprodukciju. Komanda atomski zamenjuje
postojeću izlaznu datoteku, tako da neuspešno generisanje ne ostavlja oštećen
GIF. Zbog vremenske rezolucije GIF formata, `--fps` mora deliti 100 (na primer
10, 20, 25 ili 50). Kombinacije koje bi zahtevale više od 256 MiB memorije za
frejmove odbijaju se uz jasnu poruku. Sve opcije su dostupne preko
`python animacija.py --help`.

## Struktura

- `simulacija.py` — parametri, validacija, pobudna sila i numerički rešavač
- `animacija.py` — izbor frejmova, vizuelizacija i komandna linija
- `diferencijalna-jednacina.ipynb` — interaktivna analiza rezultata
- `tests/` — regresioni testovi modela i animacije

## Testovi

```bash
uv run --locked python -m unittest discover -s tests -v
```

Testovi proveravaju fizički bilans, analitičku amplitudu ustaljenog odziva,
validaciju ulaza, vremensku osu, geometriju i vremensko uzorkovanje animacije.

## Licenca

MIT
