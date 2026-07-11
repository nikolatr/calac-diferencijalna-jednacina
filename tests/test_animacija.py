"""Testovi čistih pomoćnih funkcija animacije."""

import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

import animacija
import simulacija


class GeometrijaTest(unittest.TestCase):
    def test_opruga_pocinje_i_zavrsava_se_na_zadatim_tackama(self) -> None:
        xs, ys = animacija.spring_xy(0.0, 4.0, animacija.SPRING_Y)

        self.assertEqual(xs.size, 25)
        self.assertEqual(ys.size, 25)
        np.testing.assert_allclose([xs[0], ys[0]], [0.0, animacija.SPRING_Y])
        np.testing.assert_allclose([xs[-1], ys[-1]], [4.0, animacija.SPRING_Y])

    def test_opruga_je_prikacena_unutar_visine_mase(self) -> None:
        self.assertLessEqual(abs(animacija.SPRING_Y), animacija.MASS_H / 2)

    def test_neispravna_geometrija_opruge_se_odbacuje(self) -> None:
        with self.assertRaises(ValueError):
            animacija.spring_xy(1.0, 0.0, 0.0)
        with self.assertRaises(ValueError):
            animacija.spring_xy(0.0, 1.0, 0.0, n_coils=0)


class FrejmoviTest(unittest.TestCase):
    def test_realna_brzina_daje_25_frejmova_po_sekundi(self) -> None:
        indeksi = animacija.indeksi_frejmova(simulacija.t, fps=25, brzina=1.0)

        self.assertEqual(indeksi.size, 75)
        self.assertEqual(indeksi[0], 0)
        self.assertAlmostEqual(simulacija.t[indeksi[-1]], 2.96, places=4)

    def test_sporija_reprodukcija_daje_vise_frejmova(self) -> None:
        indeksi = animacija.indeksi_frejmova(simulacija.t, fps=25, brzina=0.5)
        self.assertEqual(indeksi.size, 150)

    def test_visok_fps_ponavlja_odbirke_umesto_da_skrati_animaciju(self) -> None:
        indeksi = animacija.indeksi_frejmova([0.0, 0.5, 1.0], fps=10)

        self.assertEqual(indeksi.size, 10)
        np.testing.assert_array_equal(indeksi, [0] * 5 + [1] * 5)

    def test_neispravni_parametri_se_odbacuju(self) -> None:
        with self.assertRaises(ValueError):
            animacija.indeksi_frejmova([0.0, 1.0], fps=0)
        with self.assertRaises(ValueError):
            animacija.indeksi_frejmova([0.0, 1.0], brzina=0)
        with self.assertRaises(ValueError):
            animacija.indeksi_frejmova([1.0, 0.0])


class KomandnaLinijaTest(unittest.TestCase):
    def test_parser_prihvata_opcije(self) -> None:
        args = animacija.napravi_parser().parse_args(
            ["--output", "izlaz.gif", "--fps", "20", "--brzina", "2", "--dpi", "60"]
        )

        self.assertEqual(args.output, "izlaz.gif")
        self.assertEqual(args.fps, 20)
        self.assertEqual(args.brzina, 2.0)
        self.assertEqual(args.dpi, 60)

    def test_neispravan_izlaz_ne_pokrece_simulaciju(self) -> None:
        with mock.patch("animacija.resi") as resi, self.assertRaises(ValueError):
            animacija.create_animation("izlaz.mp4")
        resi.assert_not_called()

    def test_gif_fps_mora_biti_tacno_predstavljiv(self) -> None:
        with mock.patch("animacija.resi") as resi, self.assertRaises(ValueError):
            animacija.create_animation("izlaz.gif", fps=30)
        resi.assert_not_called()

    def test_preveliki_bafer_se_odbacuje_pre_simulacije(self) -> None:
        with mock.patch("animacija.resi") as resi, self.assertRaisesRegex(
            ValueError, "bafera"
        ):
            animacija.create_animation("izlaz.gif", brzina=0.01)
        resi.assert_not_called()

    def test_ekstremno_mala_brzina_ne_alocira_ogroman_niz(self) -> None:
        for brzina in (1e-8, 5e-324):
            with self.subTest(brzina=brzina), mock.patch(
                "animacija.resi"
            ) as resi, self.assertRaisesRegex(ValueError, "previše frejmova"):
                animacija.create_animation("izlaz.gif", brzina=brzina)
            resi.assert_not_called()

    def test_import_ne_ucitava_pyplot(self) -> None:
        koren = Path(__file__).resolve().parents[1]
        provera = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import animacija; print('matplotlib.pyplot' in sys.modules)",
            ],
            cwd=koren,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(provera.stdout.strip(), "False")

    def test_privremena_datoteka_je_jedinstvena_i_postuje_umask(self) -> None:
        with tempfile.TemporaryDirectory() as direktorijum:
            output = Path(direktorijum) / "rezultat.gif"
            stari_umask = os.umask(0o022)
            try:
                prva = animacija._rezervisi_privremenu_putanju(output)
                druga = animacija._rezervisi_privremenu_putanju(output)
            finally:
                os.umask(stari_umask)

            self.assertNotEqual(prva, druga)
            self.assertEqual(stat.S_IMODE(prva.stat().st_mode), 0o644)
            prva.unlink()
            druga.unlink()

    def test_greska_tokom_postavke_cisti_figuru_i_privremenu_datoteku(self) -> None:
        import matplotlib.pyplot as plt

        with tempfile.TemporaryDirectory() as direktorijum:
            output = Path(direktorijum) / "rezultat.gif"
            otvorene_pre = set(plt.get_fignums())
            with mock.patch(
                "animacija.spring_xy", side_effect=RuntimeError("test greška")
            ), self.assertRaisesRegex(RuntimeError, "test greška"):
                animacija.create_animation(output, fps=2, brzina=10, dpi=20)

            self.assertEqual(set(plt.get_fignums()), otvorene_pre)
            self.assertEqual(list(Path(direktorijum).iterdir()), [])


class GenerisaniGifTest(unittest.TestCase):
    def test_praceni_gif_odgovara_podrazumevanim_podesavanjima(self) -> None:
        putanja = Path(__file__).resolve().parents[1] / "animacija.gif"
        ocekivani_frejmovi = animacija.indeksi_frejmova(
            simulacija.t,
            fps=animacija.FPS,
            brzina=animacija.BRZINA,
        ).size
        ocekivana_velicina = tuple(
            int(np.ceil(dimenzija * animacija.DPI)) for dimenzija in animacija.FIGSIZE
        )

        with Image.open(putanja) as gif:
            self.assertEqual(gif.size, ocekivana_velicina)
            self.assertEqual(gif.n_frames, ocekivani_frejmovi)
            trajanja = []
            for indeks in range(gif.n_frames):
                gif.seek(indeks)
                trajanja.append(gif.info["duration"])

        self.assertEqual(set(trajanja), {1000 // animacija.FPS})
        self.assertEqual(sum(trajanja), 3_000)


if __name__ == "__main__":
    unittest.main()
