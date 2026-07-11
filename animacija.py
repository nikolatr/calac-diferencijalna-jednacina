"""Generisanje GIF animacije sistema masa-opruga-prigušivač.

Primer::

    python animacija.py --output animacija.gif --fps 25 --brzina 1
"""

import argparse
import secrets
from math import ceil, isfinite
from pathlib import Path
from typing import Callable, Optional, Sequence, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from simulacija import PARAMETRI, resi, sila, t


SCALE = 1500
FPS = 25
BRZINA = 1.0
DPI = 80
N_COILS = 10
FIGSIZE = (10.0, 6.5)
MAX_BUFFER_BYTES = 256 * 1024 * 1024
MAX_FRAME_COUNT = 100_000

# Geometrija šeme.
WALL_X = 0.0
REST_X = 4.0
MASS_W = 1.0
MASS_H = 1.2
SPRING_Y = 0.35
DAMP_Y = -0.35
DAMP_CYL_W = 1.0
DAMP_CYL_H = 0.5


def spring_xy(
    x_start: float,
    x_end: float,
    y: float,
    n_coils: int = N_COILS,
    amp: float = 0.25,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Generiše koordinate zig-zag opruge između dve tačke."""
    if n_coils < 1:
        raise ValueError("n_coils mora biti najmanje 1")
    if amp < 0:
        raise ValueError("amp ne sme biti negativan")

    lead = min(0.3, max((x_end - x_start) / 4, 0.0))
    if x_end <= x_start or x_end - x_start <= 2 * lead:
        raise ValueError("kraj opruge mora biti desno od početka")

    broj_tacaka = 2 * n_coils + 1
    x_uvod = np.array([x_start, x_start + lead])
    x_izvod = np.array([x_end - lead, x_end])
    x_navoja = np.linspace(x_start + lead, x_end - lead, broj_tacaka)
    y_navoja = amp * np.power(-1.0, np.arange(broj_tacaka))
    xs = np.concatenate([x_uvod, x_navoja, x_izvod])
    ys = np.concatenate([[0.0, 0.0], y_navoja, [0.0, 0.0]])
    return xs, ys + y


def indeksi_frejmova(
    vreme: ArrayLike,
    fps: int = FPS,
    brzina: float = BRZINA,
) -> NDArray[np.int64]:
    """Bira odbirke tako da GIF prati zadatu brzinu reprodukcije.

    ``brzina=1`` predstavlja realno vreme, dok ``brzina=0.5`` daje dvostruko
    sporiju reprodukciju.
    """
    vreme_niz = np.asarray(vreme, dtype=float)
    if vreme_niz.ndim != 1 or vreme_niz.size < 2:
        raise ValueError("vreme mora sadržati najmanje dva odbirka")
    if not np.all(np.isfinite(vreme_niz)) or not np.all(np.diff(vreme_niz) > 0):
        raise ValueError("vreme mora biti konačno i strogo rastuće")
    if isinstance(fps, bool) or not isinstance(fps, (int, np.integer)) or fps <= 0:
        raise ValueError("fps mora biti pozitivan ceo broj")
    if not np.isfinite(brzina) or brzina <= 0:
        raise ValueError("brzina mora biti veća od nule")

    broj_frejmova = _broj_frejmova(vreme_niz, fps, brzina)
    vremena_frejmova = vreme_niz[0] + np.arange(broj_frejmova) * brzina / fps
    indeksi = np.searchsorted(vreme_niz, vremena_frejmova, side="right") - 1
    indeksi = np.clip(indeksi, 0, vreme_niz.size - 1)
    return indeksi.astype(np.int64)


def _broj_frejmova(
    vreme: NDArray[np.float64],
    fps: int,
    brzina: float,
) -> int:
    """Računa broj frejmova bez alokacije potencijalno ogromnog niza."""
    trajanje = float(vreme[-1] - vreme[0])
    zahtevano = trajanje * float(fps) / float(brzina)
    if not isfinite(zahtevano) or zahtevano > MAX_FRAME_COUNT:
        raise ValueError(
            f"animacija zahteva previše frejmova (maksimum je {MAX_FRAME_COUNT:,})"
        )
    return max(2, ceil(zahtevano))


def create_animation(
    output: Union[str, Path] = "animacija.gif",
    fps: int = FPS,
    brzina: float = BRZINA,
    dpi: int = DPI,
) -> Path:
    """Kreira animaciju, atomski je upisuje na disk i vraća njenu putanju."""
    ciscenja: list[Callable[[], None]] = []
    try:
        return _create_animation(output, fps, brzina, dpi, ciscenja)
    finally:
        for ocisti in reversed(ciscenja):
            try:
                ocisti()
            except Exception:
                # Greška pri čišćenju ne sme prikriti prvobitni ishod renderovanja.
                pass


def _create_animation(
    output: Union[str, Path],
    fps: int,
    brzina: float,
    dpi: int,
    ciscenja: list[Callable[[], None]],
) -> Path:
    """Implementira renderovanje uz registar resursa koji se uvek čiste."""
    if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi <= 0:
        raise ValueError("dpi mora biti pozitivan ceo broj")
    if isinstance(fps, bool) or not isinstance(fps, (int, np.integer)) or fps <= 0:
        raise ValueError("fps mora biti pozitivan ceo broj")
    if fps > 100 or 100 % fps != 0:
        raise ValueError("fps mora biti delilac broja 100 zbog GIF vremenske rezolucije")
    if not np.isfinite(brzina) or brzina <= 0:
        raise ValueError("brzina mora biti veća od nule")

    output_path = Path(output).expanduser()
    if output_path.suffix.lower() != ".gif":
        raise ValueError("izlazna datoteka mora imati .gif ekstenziju")

    broj_frejmova = _broj_frejmova(t, fps, brzina)
    sirina_px = int(np.ceil(FIGSIZE[0] * dpi))
    visina_px = int(np.ceil(FIGSIZE[1] * dpi))
    procenjeni_bafer = sirina_px * visina_px * 4 * broj_frejmova
    if procenjeni_bafer > MAX_BUFFER_BYTES:
        procena_mib = procenjeni_bafer / (1024 * 1024)
        maksimum_mib = MAX_BUFFER_BYTES / (1024 * 1024)
        raise ValueError(
            f"animacija bi zahtevala oko {procena_mib:.0f} MiB bafera; "
            f"smanjite --fps ili --dpi, odnosno povećajte --brzina "
            f"(ograničenje je {maksimum_mib:.0f} MiB)"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    idx = indeksi_frejmova(t, fps=fps, brzina=brzina)

    # Uvoz ovde ostavlja izbor Matplotlib backend-a pozivaocu modula.
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    rezultat = resi()
    t_sub = t[idx]
    x_sub = rezultat[idx, 0]
    f_sub = np.asarray(sila(t_sub), dtype=float)

    privremena_putanja = _rezervisi_privremenu_putanju(output_path)
    ciscenja.append(lambda: privremena_putanja.unlink(missing_ok=True))

    fig, (ax_sch, ax_x) = plt.subplots(
        2,
        1,
        figsize=FIGSIZE,
        gridspec_kw={"height_ratios": [1, 1.2]},
    )
    ciscenja.append(lambda: plt.close(fig))
    fig.subplots_adjust(hspace=0.35, right=0.87)

    # Gornji panel — šema.
    ax_sch.set_xlim(-1, 9)
    ax_sch.set_ylim(-1.8, 2.0)
    ax_sch.set_aspect("equal")
    ax_sch.axis("off")
    ax_sch.set_title("Šema sistema", fontsize=13)

    wall = plt.Rectangle(
        (-0.6, -1.5), 0.6, 3.2, fc="lightgray", ec="k", hatch="///"
    )
    ax_sch.add_patch(wall)
    ax_sch.plot([0, 0], [-1.5, 1.7], "k", lw=2)

    sx, sy = spring_xy(WALL_X, REST_X, SPRING_Y)
    (spring_line,) = ax_sch.plot(sx, sy, "k", lw=1.5)

    (rod_left,) = ax_sch.plot([], [], "k", lw=2)
    (cyl_left,) = ax_sch.plot([], [], "k", lw=2)
    (cyl_top,) = ax_sch.plot([], [], "k", lw=2)
    (cyl_bot,) = ax_sch.plot([], [], "k", lw=2)
    (piston,) = ax_sch.plot([], [], "k", lw=3)
    (rod_right,) = ax_sch.plot([], [], "k", lw=2)

    mass_patch = FancyBboxPatch(
        (REST_X, -MASS_H / 2),
        MASS_W,
        MASS_H,
        boxstyle="round,pad=0.05",
        fc="#4a90d9",
        ec="k",
        lw=2,
        zorder=5,
    )
    ax_sch.add_patch(mass_patch)
    mass_label = ax_sch.text(
        REST_X + MASS_W / 2,
        0,
        "M",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color="white",
        zorder=6,
    )

    force_arrow = FancyArrowPatch(
        (REST_X + MASS_W, 0),
        (REST_X + MASS_W + 1.5, 0),
        arrowstyle="->",
        mutation_scale=20,
        color="red",
        lw=2.5,
        zorder=5,
    )
    ax_sch.add_patch(force_arrow)
    force_text = ax_sch.text(
        REST_X + MASS_W + 1.6,
        0.35,
        "",
        fontsize=10,
        color="red",
        fontweight="bold",
        zorder=6,
    )

    time_text = ax_sch.text(
        7.5,
        1.6,
        "",
        fontsize=11,
        ha="right",
        bbox={"boxstyle": "round", "fc": "wheat", "alpha": 0.8},
    )

    ax_sch.text(
        (WALL_X + REST_X) / 2,
        SPRING_Y + 0.5,
        "opruga (c)",
        ha="center",
        fontsize=9,
        style="italic",
        color="gray",
    )
    ax_sch.text(
        (WALL_X + REST_X) / 2,
        DAMP_Y - 0.55,
        "prigušivač (μ)",
        ha="center",
        fontsize=9,
        style="italic",
        color="gray",
    )

    # Donji panel — odvojene ose zbog različitih fizičkih jedinica.
    ax_v = ax_x.twinx()
    ax_x.set_xlim(t[0], t[-1])
    x_max = max(float(np.max(np.abs(rezultat[:, 0]))) * 1.15, 1e-12)
    v_max = max(float(np.max(np.abs(rezultat[:, 1]))) * 1.15, 1e-12)
    ax_x.set_ylim(-x_max, x_max)
    ax_v.set_ylim(-v_max, v_max)
    ax_x.set_xlabel("t [s]")
    ax_x.set_ylabel("x(t) [m]", color="tab:blue")
    ax_v.set_ylabel("v(t) [m/s]", color="tab:green")
    ax_x.tick_params(axis="y", labelcolor="tab:blue")
    ax_v.tick_params(axis="y", labelcolor="tab:green")
    ax_x.set_title("Vremenski odziv", fontsize=13)
    ax_x.grid(True, alpha=0.3)

    (line_x,) = ax_x.plot([], [], color="tab:blue", lw=1.5, label="x(t)")
    (line_v,) = ax_v.plot([], [], color="tab:green", lw=1.2, label="v(t)")
    (cursor,) = ax_x.plot([], [], "r-", lw=0.8, alpha=0.6)
    ax_x.legend([line_x, line_v], ["x(t)", "v(t)"], loc="upper right")

    def update(i: int) -> list[object]:
        xi = x_sub[i]
        fi = f_sub[i]
        ti = t_sub[i]
        mass_x = REST_X + xi * SCALE

        sx, sy = spring_xy(WALL_X, mass_x, SPRING_Y)
        spring_line.set_data(sx, sy)

        cyl_left_x = (WALL_X + mass_x) / 2 - DAMP_CYL_W / 2
        cyl_right_x = cyl_left_x + DAMP_CYL_W
        dy = DAMP_Y
        hh = DAMP_CYL_H / 2
        rod_left.set_data([WALL_X, cyl_left_x], [dy, dy])
        cyl_left.set_data([cyl_left_x, cyl_left_x], [dy - hh, dy + hh])
        cyl_top.set_data([cyl_left_x, cyl_right_x], [dy + hh, dy + hh])
        cyl_bot.set_data([cyl_left_x, cyl_right_x], [dy - hh, dy - hh])
        piston_x = cyl_left_x + DAMP_CYL_W * 0.5 + (mass_x - REST_X) * 0.3
        piston_x = np.clip(piston_x, cyl_left_x + 0.05, cyl_right_x - 0.05)
        piston.set_data([piston_x, piston_x], [dy - hh + 0.03, dy + hh - 0.03])
        rod_right.set_data([piston_x, mass_x], [dy, dy])

        mass_patch.set_x(mass_x)
        mass_label.set_x(mass_x + MASS_W / 2)

        prag_sile = max(PARAMETRI.amplituda_sile * 1e-12, np.finfo(float).eps)
        if abs(fi) <= prag_sile:
            force_arrow.set_visible(False)
            force_text.set_position((mass_x + MASS_W + 0.15, 0.35))
            force_text.set_ha("left")
        else:
            force_arrow.set_visible(True)
            duzina_sile = abs(fi / PARAMETRI.amplituda_sile) * 2.0
            if fi > 0:
                pocetak_strelice = mass_x + MASS_W
                kraj_strelice = pocetak_strelice + duzina_sile
                poravnanje = "left"
                tekst_x = kraj_strelice + 0.15
                tekst_y = 0.35
            else:
                pocetak_strelice = mass_x
                kraj_strelice = pocetak_strelice - duzina_sile
                poravnanje = "center"
                tekst_x = (pocetak_strelice + kraj_strelice) / 2
                tekst_y = -1.15
            force_arrow.set_positions((pocetak_strelice, 0), (kraj_strelice, 0))
            force_text.set_position((tekst_x, tekst_y))
            force_text.set_ha(poravnanje)
        force_text.set_text(f"F={fi:.1f} N")

        time_text.set_text(f"t = {ti:.3f} s")
        poslednji_odabirak = idx[i] + 1
        line_x.set_data(t[:poslednji_odabirak], rezultat[:poslednji_odabirak, 0])
        line_v.set_data(t[:poslednji_odabirak], rezultat[:poslednji_odabirak, 1])
        cursor.set_data([ti, ti], [ax_x.get_ylim()[0], ax_x.get_ylim()[1]])
        return []

    update(0)
    anim = FuncAnimation(
        fig,
        update,
        frames=len(idx),
        interval=1000 / fps,
        blit=False,
    )

    trajanje_gifa = len(idx) / fps
    print(
        f"Čuvanje animacije ({len(idx)} frejmova, {fps} FPS, "
        f"oko {trajanje_gifa:.1f} s)..."
    )
    anim.save(privremena_putanja, writer=PillowWriter(fps=fps), dpi=dpi)
    privremena_putanja.replace(output_path)

    print(f"Sačuvano: {output_path}")
    return output_path


def _rezervisi_privremenu_putanju(output: Path) -> Path:
    """Rezerviše jedinstvenu susednu putanju uz poštovanje korisnikovog umask-a."""
    for _ in range(100):
        token = secrets.token_hex(8)
        kandidat = output.with_name(f".{output.stem}.{token}.gif")
        try:
            kandidat.open("xb").close()
        except FileExistsError:
            continue
        return kandidat
    raise FileExistsError("nije moguće rezervisati privremenu GIF datoteku")


def _pozitivan_int(vrednost: str) -> int:
    try:
        broj = int(vrednost)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("mora biti ceo broj") from exc
    if broj <= 0:
        raise argparse.ArgumentTypeError("mora biti veće od nule")
    return broj


def _pozitivan_float(vrednost: str) -> float:
    try:
        broj = float(vrednost)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("mora biti broj") from exc
    if not np.isfinite(broj) or broj <= 0:
        raise argparse.ArgumentTypeError("mora biti konačno i veće od nule")
    return broj


def _gif_fps(vrednost: str) -> int:
    broj = _pozitivan_int(vrednost)
    if broj > 100 or 100 % broj != 0:
        raise argparse.ArgumentTypeError(
            "mora biti delilac broja 100 (npr. 10, 20, 25 ili 50)"
        )
    return broj


def napravi_parser() -> argparse.ArgumentParser:
    """Pravi parser komandne linije."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="animacija.gif", help="izlazna GIF putanja")
    parser.add_argument("--fps", type=_gif_fps, default=FPS, help="frejmova u sekundi")
    parser.add_argument(
        "--brzina",
        type=_pozitivan_float,
        default=BRZINA,
        help="brzina reprodukcije; 1 znači realno vreme",
    )
    parser.add_argument("--dpi", type=_pozitivan_int, default=DPI, help="rezolucija izlaza")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> Path:
    """Ulazna tačka komandne linije."""
    parser = napravi_parser()
    args = parser.parse_args(argv)

    # CLI uvek koristi backend koji ne zahteva grafičko okruženje.
    import matplotlib

    matplotlib.use("Agg")
    try:
        return create_animation(args.output, fps=args.fps, brzina=args.brzina, dpi=args.dpi)
    except ValueError as exc:
        parser.error(str(exc))
        raise AssertionError("argparse.error uvek prekida izvršavanje") from exc


if __name__ == "__main__":
    main()
