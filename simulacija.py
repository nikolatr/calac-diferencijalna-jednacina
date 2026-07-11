"""Numerička simulacija sistema masa-opruga-prigušivač.

Modul sadrži fizičke parametre, pobudnu silu i rešavač koje dele
``animacija.py`` i Jupyter notebook. Vremenska osa je poluotvoren interval:
``T_pocetak <= t < T_kraj``.
"""

from dataclasses import dataclass, fields
from math import isfinite, pi, sqrt
from typing import Optional, Sequence, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import odeint


FloatArray = NDArray[np.float64]
ScalarOrArray = Union[np.float64, FloatArray]


@dataclass(frozen=True)
class ParametriSistema:
    """Fizički parametri linearnog oscilatora sa harmonijskom pobudom."""

    masa: float = 3.2e-2                 # kg
    krutost_opruge: float = 3e5          # N/m
    koeficijent_prigusenja: float = 1.0  # N s/m
    povrsina: float = 5e-5               # m²
    pritisak: float = 40e5               # N/m²
    frekvencija_pobude: float = 1.0      # Hz

    def __post_init__(self) -> None:
        for polje in fields(self):
            vrednost = getattr(self, polje.name)
            if not isfinite(vrednost):
                raise ValueError(f"{polje.name} mora biti konačan broj")

        if self.masa <= 0:
            raise ValueError("masa mora biti veća od nule")
        if self.krutost_opruge <= 0:
            raise ValueError("krutost_opruge mora biti veća od nule")
        if self.koeficijent_prigusenja < 0:
            raise ValueError("koeficijent_prigusenja ne sme biti negativan")
        if self.povrsina < 0:
            raise ValueError("povrsina ne sme biti negativna")
        if self.pritisak < 0:
            raise ValueError("pritisak ne sme biti negativan")
        if self.frekvencija_pobude < 0:
            raise ValueError("frekvencija_pobude ne sme biti negativna")

    @property
    def amplituda_sile(self) -> float:
        """Amplituda spoljašnje sile, u njutnima."""
        return self.povrsina * self.pritisak

    @property
    def ugaona_frekvencija_pobude(self) -> float:
        """Ugaona frekvencija pobude, u rad/s."""
        return 2 * pi * self.frekvencija_pobude

    @property
    def sopstvena_frekvencija(self) -> float:
        """Neprigušena sopstvena frekvencija sistema, u Hz."""
        return sqrt(self.krutost_opruge / self.masa) / (2 * pi)

    @property
    def faktor_prigusenja(self) -> float:
        """Bezdimenzioni faktor viskoznog prigušenja ζ."""
        imenilac = 2 * sqrt(self.masa * self.krutost_opruge)
        return self.koeficijent_prigusenja / imenilac


PARAMETRI = ParametriSistema()

# Kompatibilni nazivi koje koristi postojeći notebook.
M = PARAMETRI.masa
c = PARAMETRI.krutost_opruge
u = PARAMETRI.koeficijent_prigusenja
A = PARAMETRI.povrsina
P = PARAMETRI.pritisak

# Parametri simulacije.
y0 = (0.0, 0.0)
T_pocetak = 0.0
T_kraj = 3.0
f_odabiranja = 10_000
T_odabiranja = 1 / f_odabiranja


def vremenska_osa(
    pocetak: float = T_pocetak,
    kraj: float = T_kraj,
    frekvencija_odabiranja: float = f_odabiranja,
) -> FloatArray:
    """Vraća ravnomernu vremensku osu na poluotvorenom intervalu ``[pocetak, kraj)``."""
    if not all(isfinite(x) for x in (pocetak, kraj, frekvencija_odabiranja)):
        raise ValueError("parametri vremenske ose moraju biti konačni")
    if kraj <= pocetak:
        raise ValueError("kraj mora biti veći od početka")
    if frekvencija_odabiranja <= 0:
        raise ValueError("frekvencija_odabiranja mora biti veća od nule")

    broj_odbiraka = int(np.ceil((kraj - pocetak) * frekvencija_odabiranja))
    vreme = pocetak + np.arange(broj_odbiraka, dtype=float) / frekvencija_odabiranja
    vreme = vreme[vreme < kraj]
    if vreme.size > 1 and not np.all(np.diff(vreme) > 0):
        raise ValueError(
            "traženi korak vremenske ose je premali za preciznost pokretnog zareza"
        )
    return vreme


t = vremenska_osa()


def sila(
    vreme: ArrayLike,
    parametri: ParametriSistema = PARAMETRI,
) -> ScalarOrArray:
    """Vraća harmonijsku pobudnu silu ``A P sin(2π f t)`` u njutnima."""
    vreme_niz = np.asarray(vreme, dtype=float)
    return parametri.amplituda_sile * np.sin(
        parametri.ugaona_frekvencija_pobude * vreme_niz
    )


def jednacina(
    stanje: Sequence[float],
    vreme: float,
    parametri: ParametriSistema = PARAMETRI,
) -> FloatArray:
    """Vraća ``[x', v']`` za trenutno stanje ``[x, v]``."""
    stanje_niz = np.asarray(stanje, dtype=float)
    if stanje_niz.shape != (2,):
        raise ValueError("stanje mora sadržati tačno pomeranje i brzinu")

    x, v = stanje_niz
    ubrzanje = (
        float(sila(vreme, parametri))
        - parametri.koeficijent_prigusenja * v
        - parametri.krutost_opruge * x
    ) / parametri.masa
    return np.array([v, ubrzanje], dtype=float)


def resi(
    vreme: Optional[ArrayLike] = None,
    pocetno_stanje: ArrayLike = y0,
    parametri: ParametriSistema = PARAMETRI,
) -> FloatArray:
    """Rešava sistem i vraća kolone ``x(t)`` i ``v(t)``.

    Args:
        vreme: Strogo rastuća vremenska osa. Podrazumevano se koristi ``t``.
        pocetno_stanje: Početno pomeranje i brzina ``[x0, v0]``.
        parametri: Fizički parametri sistema.

    Raises:
        ValueError: Ako ulazni nizovi nisu konačni ili pravilnog oblika.
        RuntimeError: Ako ODE rešavač ne završi uspešno.
    """
    if not isinstance(parametri, ParametriSistema):
        raise TypeError("parametri moraju biti instanca ParametriSistema")

    vreme_niz = t if vreme is None else np.asarray(vreme, dtype=float)
    if vreme_niz.ndim != 1 or vreme_niz.size < 2:
        raise ValueError("vreme mora biti jednodimenzioni niz sa najmanje dva odbirka")
    if not np.all(np.isfinite(vreme_niz)):
        raise ValueError("vreme mora sadržati samo konačne vrednosti")
    if not np.all(np.diff(vreme_niz) > 0):
        raise ValueError("vreme mora biti strogo rastuće")

    stanje_niz = np.asarray(pocetno_stanje, dtype=float)
    if stanje_niz.shape != (2,):
        raise ValueError("pocetno_stanje mora sadržati tačno dve vrednosti")
    if not np.all(np.isfinite(stanje_niz)):
        raise ValueError("pocetno_stanje mora sadržati samo konačne vrednosti")

    rezultat, informacije = odeint(
        jednacina,
        stanje_niz,
        vreme_niz,
        args=(parametri,),
        rtol=1e-9,
        atol=(1e-12, 1e-10),
        full_output=True,
    )
    if informacije.get("message") != "Integration successful.":
        poruka = informacije.get("message", "nepoznata greška")
        raise RuntimeError(f"Numerička integracija nije uspela: {poruka}")
    if not np.all(np.isfinite(rezultat)):
        raise RuntimeError("Numerička integracija je vratila nekonačne vrednosti")

    return np.asarray(rezultat, dtype=float)


__all__ = [
    "A",
    "M",
    "P",
    "PARAMETRI",
    "ParametriSistema",
    "T_kraj",
    "T_odabiranja",
    "T_pocetak",
    "c",
    "f_odabiranja",
    "jednacina",
    "resi",
    "sila",
    "t",
    "u",
    "vremenska_osa",
    "y0",
]
